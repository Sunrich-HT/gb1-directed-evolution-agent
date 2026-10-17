#!/usr/bin/env python
"""Command line entry point: give it a sequence, get a ranked mutation proposal.

    gb1-recommend --top-k 10
    python scripts/recommend.py --top-k 10
    python scripts/recommend.py --sequence MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
    python scripts/recommend.py --labels results/v2/records_final.json --seed 302 --round 2

With no labels it ranks by the evidence prior alone - that is the round-1 situation,
where nothing has been measured yet.  Given a label file it also fits the predictor
and combines the two the way the campaign does.  Every recommendation is printed with
the evidence that paid for it and with the constraints that were checked, because a
ranked list without a reason is not a recommendation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gb1agent import evidence as evidence_mod
from gb1agent.constraints import default_constraints
from gb1agent.data import load_landscape, make_split

WT_SEQUENCE = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POSITIONS = (39, 40, 41, 54)
AA = "ACDEFGHIKLMNPQRSTVWY"
COMPONENT_ZH = {"mpnn": "反折叠分数", "packing_headroom": "堆积余量",
                "burial_charge": "埋藏电荷", "interface_bonus": "Fc 界面加成",
                "pair_volume_compensation": "41/54 体积补偿", "literature": "文献证据",
                "burial_volume": "体积项"}


def genotype(seq: str) -> str:
    return "".join(seq[p - 1] for p in POSITIONS)


def kg_reasons(kg: dict, variant: str, wt: str) -> list:
    """Why this genotype, in the graph's own words."""
    muts = {f"Mut:{wt[i]}{p}{variant[i]}" for i, p in enumerate(POSITIONS)
            if variant[i] != wt[i]}
    claims = {}
    for e in kg["edges"]:
        if e["target"] in muts and e["relation"] in ("SUPPORTS", "REFUTES", "MENTIONS"):
            claims.setdefault(e["relation"], []).append((e["source"], e["target"]))
        if e["source"] in muts and e["relation"] in ("IMPROVES", "HARMS"):
            claims.setdefault(e["relation"], []).append(
                (e["source"], f'均值差 {e.get("delta_mean", "?"):+.3f}'
                              f'（已测 {e.get("n_measured", "?")} 条）'))
        if e["source"] in muts and e["relation"] == "COMPENSATES":
            claims.setdefault("COMPENSATES", []).append((e["source"], e["target"]))
    node = {n["id"]: n for n in kg["nodes"]}
    out = []
    for rel, items in claims.items():
        for a, b in items[:2]:
            if rel in ("SUPPORTS", "REFUTES", "MENTIONS"):
                st = node.get(a, {}).get("statement", "")
                out.append(f'{rel} {b.split(":")[1]} <- {a.split(":")[1]}: {st[:90]}')
            else:
                out.append(f'{a.split(":")[1]} {rel} {b}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend the next mutations for GB1")
    ap.add_argument("--sequence", default=WT_SEQUENCE,
                    help="full 56-residue domain sequence; defaults to wild type")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--data", default="data/gb1_flip.zip",
                    help="landscape file; used to enumerate the candidate space "
                         "(and, with --reveal, to look up what the proposal scores)")
    ap.add_argument("--labels", help="records json, to fit the predictor as well")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--w-prior", type=float, default=1.0)
    ap.add_argument("--w-model", type=float, default=1.0)
    ap.add_argument("--quotas", action="store_true",
                    help="also enforce the batch composition quotas (diversity and "
                         "exploration); off by default so a short list is pure ranking")
    ap.add_argument("--reveal", action="store_true",
                    help="also print the measured fitness of the proposal (evaluation only)")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    seq = args.sequence.strip().upper()
    if len(seq) != len(WT_SEQUENCE):
        sys.exit(f"sequence must be {len(WT_SEQUENCE)} residues, got {len(seq)}")
    wt = genotype(seq)
    print(f"输入序列  长度 {len(seq)}  可变位点 {'/'.join(map(str, POSITIONS))}  "
          f"当前基因型 {wt}")
    if seq != WT_SEQUENCE:
        diff = [f"{WT_SEQUENCE[i]}{i+1}{seq[i]}" for i in range(len(seq)) if seq[i] != WT_SEQUENCE[i]]
        print(f"          与参考野生型的差异: {', '.join(diff) if diff else '无'}")

    prior = evidence_mod.load("knowledge/structure_evidence.json",
                              "config/mpnn_log_odds.json", "knowledge/evidence_base.json")
    struct = json.loads(Path("knowledge/structure_evidence.json").read_text())
    kg_path = Path("knowledge/knowledge_graph.json")
    kg = json.loads(kg_path.read_text()) if kg_path.exists() else {"nodes": [], "edges": []}

    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    score = prior.score_variants(variants)
    source = "证据先验（未使用任何标签）"
    known_set = set()

    if args.labels:
        from gb1agent.predictors import FitnessModel, select_model
        fitness = land["fitness"].to_numpy(float)
        index = {v: i for i, v in enumerate(variants)}
        split = make_split(land, seed=args.seed)
        records = json.loads(Path(args.labels).read_text())
        known = list(split.initial)
        for r in records:
            if r["seed"] == args.seed and r["round"] < args.round:
                known += [index[d["variant"]] for d in r["selected"]]
        known = sorted(set(known))
        chosen = select_model(variants[known], fitness[known], seed=args.seed)
        spec = chosen[0] if isinstance(chosen, tuple) else chosen
        model = FitnessModel(spec, args.seed).fit(variants[known], fitness[known])
        pred = model.predict(variants)
        z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        score = args.w_prior * z(score) + args.w_model * z(pred)
        known_set = {str(variants[i]) for i in known}
        source = (f"{args.w_prior:g} x 证据先验 + {args.w_model:g} x 预测器"
                  f"（{len(known)} 条已测标签，{spec.kind}[{spec.encoding}]）")

    cs = default_constraints(struct)
    if not args.quotas:
        for cid in ("C5_batch_diversity", "C6_exploration_quota"):
            if cid in cs.constraints:
                cs.constraints[cid].active = False
    sel, audit = cs.select(variants, score, args.top_k,
                           rng=np.random.default_rng(args.seed), known=known_set)
    picks = [str(variants[i]) for i in sel]

    print(f"打分来源  {source}")
    print(f"配额      {'启用（多样性 + 探索）' if args.quotas else '关闭（纯排序，--quotas 可启用）'}")
    print(f"约束引擎  合法候选 {audit['n_legal_after_filters']:,} / {audit['n_pool']:,}"
          f"（{'；'.join(f'{k} 删除 {v:,}' for k, v in audit['filters_removed'].items() if v) or '无硬过滤删除'}）")
    print(f"\n推荐的前 {len(picks)} 个突变方案：\n")
    rows = []
    for rank, v in enumerate(picks, 1):
        att = prior.attribution(v)
        muts = [f"{wt[i]}{p}{v[i]}" for i, p in enumerate(POSITIONS) if v[i] != wt[i]]
        top = sorted(((k, val) for k, val in att.items() if k != "total"),
                     key=lambda kv: -abs(kv[1]))[:3]
        print(f"{rank:2d}. {v}   突变 {', '.join(muts) if muts else '（野生型）'}")
        print(f"      先验合计 {att['total']:+.2f}  主要来自 "
              + "、".join(f"{COMPONENT_ZH.get(k, k)} {val:+.2f}" for k, val in top))
        for line in kg_reasons(kg, v, wt)[:2]:
            print(f"      知识图谱: {line}")
        if args.reveal:
            f = float(land["fitness"].to_numpy(float)[list(variants).index(v)])
            print(f"      [评估用] 实测适应度 {f:.3f}")
        rows.append({"rank": rank, "variant": v, "mutations": muts,
                     "prior_total": round(att["total"], 3),
                     "attribution": {k: round(val, 3) for k, val in att.items()}})

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"input_sequence": seq, "genotype": wt, "score_source": source,
             "constraint_audit": audit, "recommendations": rows}, indent=1))
        print(f"\n已写入 {args.json_out}")
