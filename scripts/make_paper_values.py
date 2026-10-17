#!/usr/bin/env python
"""Generate paper/generated/values.tex and the result tables from results/.

No number in the paper is typed by hand: main.tex consumes macros defined here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ARMS = ["random", "greedy", "greedy_explore", "llm_agent", "llm_knowledge",
        "prior_fixed", "prior_constraint"]
LABEL = {
    "random": "随机对照",
    "greedy": "预测器贪心",
    "greedy_explore": "预测器 + 25\\% 探索",
    "llm_agent": "LLM 智能体",
    "llm_knowledge": "LLM + 知识块",
    "prior_fixed": "证据先验 + 约束（无 LLM）",
    "prior_constraint": "证据先验 + 约束（LLM 调参）",
}
KEY = {"random": "Random", "greedy": "Greedy", "greedy_explore": "GreedyExplore",
       "llm_agent": "LLMAgent", "llm_knowledge": "LLMKnowledge",
       "prior_fixed": "PriorFixed", "prior_constraint": "PriorLLM"}


def tex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--prior-eval", default="results/prior_evaluation.json")
    ap.add_argument("--interpret", default="results/interpretability.json")
    ap.add_argument("--structure", default="knowledge/structure_evidence.json")
    ap.add_argument("--evidence", default="knowledge/evidence_base.json")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--structure-analysis", default="results/structure/structure_analysis.json")
    ap.add_argument("--ledgers", nargs="+",
                    default=sorted(str(q) for q in Path("results").glob("**/llm_ledger.json")))
    ap.add_argument("--out", default="paper/generated")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.records).read_text())
    prior_eval = json.loads(Path(args.prior_eval).read_text())
    interp = json.loads(Path(args.interpret).read_text())
    struct = json.loads(Path(args.structure).read_text())
    evidence = json.loads(Path(args.evidence).read_text())
    base = json.loads(Path(args.baselines).read_text())
    sa = json.loads(Path(args.structure_analysis).read_text()) if Path(args.structure_analysis).exists() else None

    seeds = sorted({r["seed"] for r in records})
    rounds = sorted({r["round"] for r in records})
    opt = prior_eval["global_optimum"]["fitness"]
    final = {a: np.array([next(r["best_so_far"] for r in records
                               if r["arm"] == a and r["seed"] == s and r["round"] == max(rounds))
                          for s in seeds]) for a in ARMS if any(r["arm"] == a for r in records)}

    macros: list[str] = []

    def mac(name: str, value) -> None:
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    mac("Nseeds", len(seeds))
    mac("Nrounds", max(rounds))
    mac("Batch", len(records[0]["selected"]))
    mac("Ninitial", 192)
    mac("Npool", 139169)
    mac("Ntest", 10000)
    mac("Nlandscape", f"{prior_eval['n_variants']:,}".replace(",", ","))
    mac("GlobalOpt", f"{opt:.3f}")
    mac("GlobalOptVariant", prior_eval["global_optimum"]["variant"])
    mac("TotalLabels", 192 + max(rounds) * len(records[0]["selected"]))

    for arm, vals in final.items():
        k = KEY[arm]
        mac(f"Mean{k}", f"{vals.mean():.3f}")
        mac(f"Median{k}", f"{np.median(vals):.3f}")
        mac(f"Hits{k}", int((vals >= opt - 1e-6).sum()))
        mac(f"HitPct{k}", f"{100 * (vals >= opt - 1e-6).mean():.0f}")
        div = np.mean([r["diversity"]["mean_pairwise_hamming"] for r in records if r["arm"] == arm])
        mac(f"Div{k}", f"{div:.2f}")

    def paired(a: str, b: str, tag: str) -> None:
        d = final[a] - final[b]
        nz = d[np.abs(d) > 1e-9]
        mac(f"Diff{tag}", f"{d.mean():+.3f}")
        mac(f"Wins{tag}", int((d > 1e-9).sum()))
        mac(f"Losses{tag}", int((d < -1e-9).sum()))
        mac(f"Ties{tag}", int((np.abs(d) <= 1e-9).sum()))
        mac(f"Wilcoxon{tag}", f"{wilcoxon(nz).pvalue:.3f}" if len(nz) >= 5 else "n.a.")
        hit_a, hit_b = final[a] >= opt - 1e-6, final[b] >= opt - 1e-6
        nb, nc = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
        mac(f"McNemarB{tag}", nb)
        mac(f"McNemarC{tag}", nc)
        mac(f"McNemarP{tag}", f"{binomtest(nb, nb + nc, 0.5).pvalue:.3f}" if nb + nc else "n.a.")

    paired("prior_constraint", "greedy", "PriorVsGreedy")
    paired("prior_fixed", "greedy", "FixedVsGreedy")
    paired("prior_constraint", "prior_fixed", "LLMVsFixed")
    paired("llm_agent", "greedy", "LLMVsGreedy")

    # prior quality
    for key, tag in [("evidence_prior", "Evidence"), ("leakage_measured_single_mutants", "Leakage"),
                     ("only_mpnn", "MPNNonly"), ("only_literature", "LitOnly"),
                     ("only_packing_headroom", "PackOnly"), ("reference_random", "RandomPrior"),
                     ("with_symmetric_volume", "SymVol"), ("with_asymmetric_volume", "AsymVol"),
                     ("ablate_mpnn", "AblMPNN"), ("ablate_literature", "AblLit")]:
        m = prior_eval["priors"].get(key)
        if not m:
            continue
        mac(f"Rho{tag}", f"{m['spearman_all']:+.3f}")
        mac(f"RhoHD{tag}", f"{m['spearman_hd3plus']:+.3f}")
        mac(f"Top{tag}", f"{m['top5pct_precision']:.3f}")
        mac(f"Best{tag}", f"{m['best_in_top96']:.3f}")
        mac(f"Enrich{tag}", f"{m['enrichment_top1pct']:.1f}")

    # structure facts.  LaTeX control sequences cannot contain digits, so positions are
    # spelled out: \Sasathirtynine, \Buriedforty, \Freefortyone, ...
    WORD = {39: "thirtynine", 40: "forty", 41: "fortyone", 54: "fiftyfour"}
    for p, w in WORD.items():
        site = struct["sites"][str(p)]
        mac(f"Sasa{w}", f"{100 * site['rel_sasa']:.1f}")
        mac(f"Buried{w}", f"{site['interface']['buried_surface_on_binding_A2']:.1f}")
        mac(f"Phi{w}", f"{site['phi']:.1f}")
        mac(f"Psi{w}", f"{site['psi']:.1f}")
        mac(f"Free{w}", f"{site['packing_headroom']['free_radius']:.2f}")
    mac("NMRPsiFortyOne", f"{struct['nmr_crosscheck_2gb1']['41']['psi']:.1f}")

    # evidence base
    counts = evidence.get("counts") or {}
    mac("NEvidence", len(evidence["evidence"]))
    mac("NSources", len(evidence.get("sources", [])))
    mac("NPatentRows", sum(1 for r in evidence["evidence"]
                           if (r.get("source") or {}).get("type") == "patent"))

    # baselines
    d0 = {}
    for row in base["d0_models"]:
        d0.setdefault(row["spec"], []).append(row)
    best_spec = max(d0.items(), key=lambda kv: np.mean([r["spearman"] for r in kv[1]]))
    mac("BaseSpearman", f"{np.mean([r['spearman'] for r in best_spec[1]]):.3f}")
    mac("BaseTopFive", f"{np.mean([r['top5pct_precision'] for r in best_spec[1]]):.3f}")
    flip = {r["split"]: r for r in base["flip_splits"]}
    if "sampled" in flip:
        mac("FlipSampledRho", f"{flip['sampled']['spearman']:.3f}")
        mac("FlipSampledRecall", f"{flip['sampled']['top100_recall']:.2f}")
        mac("FlipSampledTrain", flip["sampled"]["n_train"])

    # interpretability
    mac("EpistasisShare", f"{100 * interp['model_layer']['epistasis_d0']['pairwise_share']:.0f}")
    verdicts = {}
    for r in records:
        for h in r["hypotheses"]:
            verdicts[h["verdict"]] = verdicts.get(h["verdict"], 0) + 1
    mac("HypTotal", sum(verdicts.values()))
    for v in ("supported", "refuted", "undetermined"):
        mac(f"Hyp{v.capitalize()}", verdicts.get(v, 0))
        mac(f"Hyp{v}", verdicts.get(v, 0))

    # structure services
    if sa:
        mac("BoltzN", sa["n_folded"])
        mac("BoltzKeys", sa["n_keys_rotated"])
        mac("BoltzRho", f"{sa['spearman_vs_fitness']['plddt_sites']:+.2f}")
        mac("BoltzSeqRho", f"{sa['spearman_vs_fitness']['sequence_model']:+.2f}")
        mac("BoltzDeltaRho", f"{sa['ablation']['delta_spearman']:+.3f}")
        mac("BoltzDeltaTop", f"{sa['ablation']['delta_top5pct']:+.3f}")
        mac("BoltzLatency", f"{sa['median_latency_s']:.1f}")

    # cost
    calls = spend = 0
    for led in args.ledgers:
        if Path(led).exists():
            d = json.loads(Path(led).read_text())
            calls += d["n_calls"]
            spend += d["spent_usd"]
    mac("LLMCalls", calls)
    mac("LLMSpend", f"{spend:.3f}")

    (out / "values.tex").write_text("\n".join(macros) + "\n")

    # main results table
    rows = []
    for arm in ARMS:
        if arm not in final:
            continue
        vals = final[arm]
        gains = [sum(r["metrics"]["gain"] for r in records if r["arm"] == arm and r["seed"] == s)
                 for s in seeds]
        div = np.mean([r["diversity"]["mean_pairwise_hamming"] for r in records if r["arm"] == arm])
        rows.append(f"{LABEL[arm]} & {vals.mean():.2f} & {np.median(vals):.2f} & "
                    f"{vals.min():.2f}–{vals.max():.2f} & "
                    f"{int((vals >= opt - 1e-6).sum())}/{len(seeds)} & {np.mean(gains):.2f} & {div:.2f} \\\\")
    table = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
             "方法 & 末轮最好（均值） & 中位数 & 范围 & 命中全局最优 & 累计收益 & 批次多样性 \\\\\n"
             "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "table_arms.tex").write_text("\\resizebox{\\textwidth}{!}{%\n" + table + "}\n")

    # prior table
    prows = []
    for key, label in [("reference_random", "随机排序"),
                       ("only_mpnn", "仅反折叠（ProteinMPNN）"),
                       ("only_packing_headroom", "仅堆积余量"),
                       ("only_literature", "仅文献条目"),
                       ("evidence_prior", "完整证据先验"),
                       ("with_symmetric_volume", "证据先验 + 对称体积项"),
                       ("leakage_measured_single_mutants", "实测单突变加性先验$^{*}$")]:
        m = prior_eval["priors"].get(key)
        if not m:
            continue
        prows.append(f"{label} & {m['spearman_all']:+.3f} & {m['spearman_hd3plus']:+.3f} & "
                     f"{m['top5pct_precision']:.3f} & {m['best_in_top96']:.2f} & "
                     f"{m['enrichment_top1pct']:.1f}$\\times$ \\\\")
    ptable = ("\\begin{tabular}{lccccc}\n\\toprule\n"
              "先验 & $\\rho$（全景） & $\\rho$（$\\geq$3 突变） & top-5\\% 精度 & 前 96 内最好 & top-1\\% 富集 \\\\\n"
              "\\midrule\n" + "\n".join(prows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "table_priors.tex").write_text("\\resizebox{\\textwidth}{!}{%\n" + ptable + "}\n")

    print(f"wrote {len(macros)} macros, 2 tables to {out}")


if __name__ == "__main__":
    main()
