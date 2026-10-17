#!/usr/bin/env python
"""Paired comparisons between arms: Wilcoxon on the final best-so-far, McNemar
on whether the arm reached the global optimum.

Both tests pair the arms seed by seed, which is the only honest comparison here:
every arm starts from the same 192 labels and sees the same candidate pool, so
the seed is the experimental unit. Writes results/significance.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

PAIRS = [("prior_constraint", "greedy"), ("prior_fixed", "greedy"),
         ("prior_constraint", "prior_fixed"), ("llm_agent", "greedy")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--prior-eval", default="results/prior_evaluation.json")
    ap.add_argument("--out", default="results/significance.json")
    args = ap.parse_args()

    records = json.loads(Path(args.records).read_text())
    opt = json.loads(Path(args.prior_eval).read_text())["global_optimum"]["fitness"]
    seeds = sorted({r["seed"] for r in records})
    last = max(r["round"] for r in records)
    final = {a: np.array([next(r["best_so_far"] for r in records
                               if r["arm"] == a and r["seed"] == s and r["round"] == last)
                          for s in seeds])
             for a in sorted({r["arm"] for r in records})}

    out = {"n_seeds": len(seeds), "rounds": last, "global_optimum": opt, "comparisons": {}}
    for a, b in PAIRS:
        if a not in final or b not in final:
            continue
        d = final[a] - final[b]
        nz = d[np.abs(d) > 1e-9]
        hit_a, hit_b = final[a] >= opt - 1e-6, final[b] >= opt - 1e-6
        nb, nc = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
        out["comparisons"][f"{a}_vs_{b}"] = {
            "mean_difference": float(d.mean()),
            "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
            "ties": int((np.abs(d) <= 1e-9).sum()),
            "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) >= 5 else None,
            "hits_a": int(hit_a.sum()), "hits_b": int(hit_b.sum()),
            "mcnemar_b": nb, "mcnemar_c": nc,
            "mcnemar_p": float(binomtest(nb, nb + nc, 0.5).pvalue) if nb + nc else None,
        }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["comparisons"].items():
        print(f"{k}: diff {v['mean_difference']:+.3f}, Wilcoxon p "
              f"{v['wilcoxon_p'] if v['wilcoxon_p'] is None else round(v['wilcoxon_p'], 3)}, "
              f"hits {v['hits_a']}/{out['n_seeds']} vs {v['hits_b']}/{out['n_seeds']}, "
              f"McNemar p {v['mcnemar_p'] if v['mcnemar_p'] is None else round(v['mcnemar_p'], 3)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
