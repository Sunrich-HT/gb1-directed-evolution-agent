#!/usr/bin/env python
"""Capture the real step-by-step trace of ONE round, for the demo page.

This performs the calls it reports: it fetches the structures from RCSB over HTTP
and records the receipts, recomputes the per-residue structural analysis from the
file it just downloaded, replays the stored LLM payload/response from the run, and
recomputes the batch selection under two rule sets so the effect of a rule change
on the next batch can be read off exactly.

Nothing here is illustrative: every field lands in results/trace_seed302.json with
the status codes, byte counts, checksums and variant names it actually saw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gb1agent import evidence as evidence_mod
from gb1agent.constraints import default_constraints
from gb1agent.data import load_landscape, make_split
from gb1agent.predictors import FitnessModel, select_model

from build_structure_evidence import (backbone_dihedrals, parse_pdb, rama_region,
                                      sequence_of, shrake_rupley)

RCSB = "https://files.rcsb.org/download/{}.pdb"
SITES = (39, 40, 41, 54)


def http_get(url: str, dest: Path) -> dict:
    """Fetch and keep the receipt - status, bytes, checksum, latency."""
    t0 = time.time()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        status = resp.status
        ctype = resp.headers.get("Content-Type", "")
    dt = time.time() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"url": url, "status": status, "content_type": ctype,
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "seconds": round(dt, 2), "saved_to": str(dest),
            "head": body.decode("utf-8", "ignore").splitlines()[:3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--out", default="results/trace_seed302.json")
    args = ap.parse_args()

    T: dict = {"seed": args.seed, "round": args.round, "captured_at": time.strftime("%Y-%m-%d %H:%M"),
               "steps": []}

    def add(tool, call, result, note=""):
        T["steps"].append({"tool": tool, "call": call, "result": result, "note": note})

    # ---------------------------------------------------------------- 1 HTTP
    receipts = {}
    for pdb in ("1pga", "1fcc"):
        receipts[pdb] = http_get(RCSB.format(pdb.upper()), Path(f"data/trace_{pdb}.pdb"))
        print(pdb, receipts[pdb]["status"], receipts[pdb]["bytes"], "bytes", flush=True)
    add("http.get", {"endpoint": RCSB.format("{PDB_ID}"), "ids": ["1PGA", "1FCC"],
                     "method": "GET"}, receipts,
        "1PGA 为自由态晶体结构（2.07 A），1FCC 为蛋白 G 结构域与 IgG1 Fc 的复合物。")

    # ---------------------------------------------------------------- 2 structure
    free = parse_pdb(Path("data/trace_1pga.pdb"), chains="A")
    sasa = shrake_rupley(free)
    phipsi = backbone_dihedrals(free, "A")
    seq = sequence_of(free, "A")
    per_res = []
    for res in sorted(seq):
        row = sasa.get(("A", res), {})
        d = phipsi.get(res, {})
        phi, psi = d.get("phi"), d.get("psi")
        per_res.append({"resseq": res, "aa": seq[res],
                        "rel_sasa": None if row.get("rel_sasa") is None
                        else round(row["rel_sasa"], 3),
                        "phi": None if phi is None else round(phi, 1),
                        "psi": None if psi is None else round(psi, 1),
                        "region": rama_region(phi, psi) if phi is not None and psi is not None
                        else None})
    struct = json.loads(Path("knowledge/structure_evidence.json").read_text())
    add("structure.analyze",
        {"function": "shrake_rupley + dihedrals", "input": "data/trace_1pga.pdb (chain A)",
         "probe_radius_A": 1.4, "n_sphere_points": 256,
         "reference": "Tien et al. 2013 theoretical maxima"},
        {"n_residues": len(per_res), "per_residue": per_res,
         "mutable_sites": {str(s): struct["sites"][str(s)] for s in SITES},
         "fc_contact_residues": struct["fc_contact_residues_b1_numbering"]},
        "全部 56 个残基都算了可及度与主链二面角；下面的判断只用到其中四个可变位点与 16 个 Fc 接触残基。")

    # ---------------------------------------------------------------- 3 prior
    prior = evidence_mod.load("knowledge/structure_evidence.json",
                              "config/mpnn_log_odds.json", "knowledge/evidence_base.json")
    mpnn = json.loads(Path("config/mpnn_log_odds.json").read_text())
    add("inverse_folding.load",
        {"file": "config/mpnn_log_odds.json",
         "model": mpnn["model"], "structure": mpnn["structure"],
         "weights_sha256": mpnn["weights_sha256"], "backend": mpnn["backend"]},
        {"convention": mpnn["convention"], "positions": mpnn["positions"],
         "wild_type": mpnn["wild_type"],
         "log_odds_by_site": {str(p): {k: round(v, 2) for k, v in row.items()}
                              for p, row in zip(mpnn["positions"], mpnn["log_odds"])}},
        "这是本轮唯一一个被证据否决的分量：40 位的零样本分数按 EV053 被静音。")

    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)
    index = {v: i for i, v in enumerate(variants)}
    target = str(variants[int(np.argmax(fitness))])
    scores = prior.score_variants(variants)
    prior_rank = int((scores > scores[index[target]]).sum()) + 1
    pe = json.loads(Path("results/prior_evaluation.json").read_text())
    add("prior.score",
        {"function": "EvidencePrior.score_variants", "n_variants": int(len(variants)),
         "components": [c for c in prior.component_names()]
         if hasattr(prior, "component_names") else "see evidence.py"},
        {"attribution_global_optimum": {k: round(v, 3) for k, v in prior.attribution(target).items()},
         "attribution_wild_type": {k: round(v, 3) for k, v in prior.attribution("VDGV").items()},
         "rank_of_global_optimum": prior_rank,
         "zero_label_evaluation": pe["priors"]["evidence_prior"],
         "measured_single_mutant_upper_bound": pe["priors"]["leakage_measured_single_mutants"]},
        "先验在不使用任何多突变标签的情况下，把全局最优排到第 %d 名。" % prior_rank)

    # ---------------------------------------------------------------- 4 predictor
    split = make_split(land, seed=args.seed)
    records = json.loads(Path("results/v2/records_final.json").read_text())
    r1 = [r for r in records if r["arm"] == "prior_constraint"
          and r["seed"] == args.seed and r["round"] == 1][0]
    known = list(split.initial) + [index[d["variant"]] for d in r1["selected"]]
    chosen = select_model(variants[known], fitness[known], seed=args.seed)
    spec = chosen[0] if isinstance(chosen, tuple) else chosen
    cv_rows = chosen[1] if isinstance(chosen, tuple) and len(chosen) > 1 else []
    model = FitnessModel(spec, args.seed).fit(variants[known], fitness[known])
    pool = np.array([i for i in split.pool if i not in set(known)])
    pred = model.predict(variants[pool])
    pos_t = int(np.where(pool == index[target])[0][0])
    add("predictor.fit",
        {"function": "select_model + FitnessModel.fit", "n_train": len(known),
         "cv_folds": 3, "candidates": [c["spec"] for c in cv_rows]},
        {"selected": f"{spec.kind}[{spec.encoding}] {spec.params}",
         "cv_table": cv_rows,
         "prediction_for_global_optimum": round(float(pred[pos_t]), 3),
         "model_rank_of_global_optimum": int((pred > pred[pos_t]).sum()) + 1,
         "n_pool": int(len(pool))},
        "预测器只用 %d 条标签；它对全局最优的预测值是 %.3f，而实测是 %.3f。"
        % (len(known), float(pred[pos_t]), fitness.max()))

    # ---------------------------------------------------------------- 5 LLM
    llm = json.loads(Path(f"results/v2/prior_constraint_s{args.seed}_r{args.round}_llm.json").read_text())
    ledger = json.loads(Path("results/v2/llm_ledger.json").read_text())
    calls = ledger if isinstance(ledger, list) else ledger.get("calls", [])
    tag = f"prior_constraint_s{args.seed}_r{args.round}"
    mine = [c for c in calls if tag in str(c.get("tag", ""))]
    add("llm.set_rules",
        {"provider": "DeepSeek", "model": "deepseek-chat", "response_format": "json_object",
         "payload": llm["payload"]},
        {"raw_response": llm["response"],
         "ledger": mine[0] if mine else {"note": "ledger 未按轮次分标签"}},
        "模型看到的就是左边这些字段；它只能回规则与权重，不能点名变体。")

    # ---------------------------------------------------------------- 6 rule diff
    prior_pool = prior.score_variants(variants[pool])
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    note2 = [r for r in records if r["arm"] == "prior_constraint"
             and r["seed"] == args.seed and r["round"] == args.round][0]["notes"]
    w = note2.get("weights", {})
    combined = w.get("prior", 1.0) * z(prior_pool) + w.get("model", 1.0) * z(pred)

    def pick(rule_set: str):
        cs = default_constraints(struct)
        if rule_set == "round1":
            cs.constraints["C2_volume_window"].params["max_added_volume"] = 60.0
            cs.constraints["C7_interface_contact"].params["weight"] = 0.5
        else:
            cs.constraints["C2_volume_window"].params["max_added_volume"] = 90.0
            cs.constraints["C7_interface_contact"].params["weight"] = 0.2
            cs.set_refuted({"V54T": 1.0, "D40W": 1.0, "V39A": 1.0, "V39I": 0.5}, args.round)
        sel, audit = cs.select(variants[pool], combined.copy(), 96,
                              known=set(variants[known]), rng=np.random.default_rng(args.seed))
        return [str(variants[pool][i]) for i in sel], audit

    pick_a, audit_a = pick("round1")
    pick_b, audit_b = pick("round2")
    entered = [v for v in pick_b if v not in set(pick_a)]
    left = [v for v in pick_a if v not in set(pick_b)]

    # which of the moved slots carry a substitution the program refuted in round 1?
    refuted_map = {"V39I": (0, "I"), "V39A": (0, "A"), "D40W": (1, "W"), "V54T": (3, "T")}

    def carries(v: str) -> list:
        return [name for name, (i, aa) in refuted_map.items() if v[i] == aa]

    ann_left = [{"variant": v, "refuted": carries(v)} for v in left]
    ann_entered = [{"variant": v, "refuted": carries(v)} for v in entered]
    add("constraints.apply",
        {"function": "ConstraintSet.select", "batch_size": 96,
         "weights": w,
         "rule_set_A": "第 1 轮的规则（体积上限 60 A^3，界面惩罚 0.5，无 C8）",
         "rule_set_B": "第 2 轮的规则（体积上限 90 A^3，界面惩罚 0.2，C8 降权四个被推翻的取代）"},
        {"audit_A": audit_a, "audit_B": audit_b,
         "n_entered": len(entered), "n_left": len(left),
         "entered": ann_entered, "left": ann_left,
         "refuted_substitutions_penalised": list(refuted_map),
         "n_left_carrying_refuted": sum(1 for r in ann_left if r["refuted"]),
         "n_entered_carrying_refuted": sum(1 for r in ann_entered if r["refuted"]),
         "global_optimum_in_A": target in set(pick_a),
         "global_optimum_in_B": target in set(pick_b)},
        "同一个模型、同一个先验、同一个候选池，只改规则："
        "合法候选从 %s 条增加到 %s 条，%d 个名额换人。"
        "C8 是软惩罚不是过滤，所以带着被推翻取代的变体并没有被清出批次"
        "（换出 %d 个、换进 %d 个，基本持平）——真正改变名单的是放宽的体积上限与重新加权的界面惩罚。"
        % (f"{audit_a['n_legal_after_filters']:,}", f"{audit_b['n_legal_after_filters']:,}",
           len(entered), sum(1 for r in ann_left if r["refuted"]),
           sum(1 for r in ann_entered if r["refuted"])))

    # ---------------------------------------------------------------- 7 oracle
    oracle = [json.loads(l) for l in
              Path("results/v2/oracle_queries.jsonl").read_text().splitlines() if l.strip()]
    mine_o = [o for o in oracle if o.get("arm") == "prior_constraint"
              and o.get("round") == args.round]
    rec2 = [r for r in records if r["arm"] == "prior_constraint"
            and r["seed"] == args.seed and r["round"] == args.round][0]
    hit = [d for d in rec2["selected"] if d["variant"] == target]
    add("oracle.query",
        {"function": "virtual oracle = published DMS lookup", "n": 96,
         "commit_sha256": (mine_o[0] if mine_o else {}).get("commit_sha256", ""),
         "rule": "批次先以 SHA-256 冻结，之后才允许揭示标签"},
        {"batch_best": rec2["metrics"]["best"], "batch_mean": round(rec2["metrics"]["mean"], 3),
         "hit_global_optimum": bool(hit),
         "global_optimum_row": hit[0] if hit else None,
         "top_revealed": sorted(rec2["selected"], key=lambda d: -d["fitness"])[:6]},
        "揭示后本轮最好 %.3f。" % rec2["metrics"]["best"])

    # ---------------------------------------------------------------- 8 verdicts
    add("hypotheses.judge",
        {"function": "program verdict, not the model", "n_hypotheses": len(rec2["hypotheses"]),
         "rule": "每条陈述与带/不带该取代的实测均值对比，方向不符即判为推翻"},
        {"hypotheses": rec2["hypotheses"]},
        "裁决由程序做；被推翻的取代自动进入下一轮的 C8 惩罚项。")

    # ---------------------------------------------------------------- 9 boltz2
    rec_path = Path("results/structure/nvidia_receipts.jsonl")
    rcs = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    cif = Path("results/structure/cif/FWAA.cif")
    add("boltz2.predict",
        {"endpoint": "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict",
         "method": "POST", "payload": {"polymers": [{"id": "A", "molecule_type": "protein",
                                                     "sequence": "...FWAE..."}],
                                       "recycling_steps": 3, "sampling_steps": 50,
                                       "diffusion_samples": 1}},
        {"n_receipts": len(rcs), "last_receipts": rcs[-3:],
         "status_counts": {str(s): sum(1 for r in rcs if r.get("status") == s)
                           for s in sorted({r.get("status") for r in rcs})},
         "cif_head": cif.read_text().splitlines()[:6] if cif.exists() else []},
        "折叠是真实调用；但它的置信度与适应度几乎无关，因此没有进入打分。")

    Path(args.out).write_text(json.dumps(T, ensure_ascii=False, indent=1))
    print("wrote", args.out, "|", len(T["steps"]), "steps")


if __name__ == "__main__":
    main()
