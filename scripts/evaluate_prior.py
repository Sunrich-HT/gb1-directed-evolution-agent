#!/usr/bin/env python
"""How good is the evidence prior, before a single multi-mutant label is seen?

Everything here is a *no-label* evaluation: a prior is a ranking of the 149,361
variants built only from structure, biophysics and published claims, and the
measured landscape is used once, at the end, to score that ranking.

Reported per prior:
  spearman            over the whole landscape and over the >=3-mutation subset
  top5pct_precision   of the prior's top 5 %, how many are in the true top 5 %
  best_in_top96       the best true fitness inside one 96-variant batch
  enrichment_top1pct  P(true top 1 % | prior top 96) / base rate
  recall_top100       how many of the 100 fittest variants fall in the prior's top 1 %

Ablations drop one component at a time, which is what turns "the prior helped"
into a statement with a number attached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent import evidence as evidence_mod
from gb1agent.data import load_landscape
from gb1agent.priors import load_log_odds, score_variants, single_mutant_table, zscore_by_substitution


def evaluate(scores: np.ndarray, fitness: np.ndarray, n_mut: np.ndarray, batch: int = 96) -> dict:
    order = np.argsort(-scores)
    n = len(scores)
    top5_pred = order[: max(1, n // 20)]
    true_top5 = set(np.argsort(-fitness)[: max(1, n // 20)].tolist())
    true_top1 = set(np.argsort(-fitness)[: max(1, n // 100)].tolist())
    true_top100 = set(np.argsort(-fitness)[:100].tolist())
    head96 = order[:batch]
    base_rate = len(true_top1) / n
    hits = sum(1 for i in head96 if i in true_top1) / batch
    deep = n_mut >= 3
    return {
        "spearman_all": float(spearmanr(scores, fitness).statistic),
        "spearman_hd3plus": float(spearmanr(scores[deep], fitness[deep]).statistic),
        "top5pct_precision": float(np.mean([i in true_top5 for i in top5_pred])),
        "best_in_top96": float(fitness[head96].max()),
        "enrichment_top1pct": float(hits / base_rate) if base_rate else float("nan"),
        "recall_top100_in_prior_top1pct": float(
            len(true_top100 & set(order[: max(1, n // 100)].tolist())) / 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--structure-evidence", default="knowledge/structure_evidence.json")
    ap.add_argument("--evidence-base", default="knowledge/evidence_base.json")
    ap.add_argument("--mpnn", default="config/mpnn_log_odds.json")
    ap.add_argument("--out", default="results/prior_evaluation.json")
    args = ap.parse_args()

    df = load_landscape(args.data)
    variants = df["variant"].to_numpy()
    fitness = df["fitness"].to_numpy(float)
    n_mut = df["n_mut"].to_numpy(int)

    priors: dict[str, np.ndarray] = {}
    detail: dict[str, dict] = {}

    full = evidence_mod.load(args.structure_evidence, args.mpnn, args.evidence_base)
    priors["evidence_prior"] = full.score_variants(variants)
    detail["evidence_prior"] = {"components": [c.name for c in full.components],
                                "weights": full.weights}

    # leave-one-component-out ablations
    for comp in [c.name for c in full.components]:
        w = dict(full.weights)
        w[comp] = 0.0
        ab = evidence_mod.load(args.structure_evidence, args.mpnn, args.evidence_base, weights=w)
        priors[f"ablate_{comp}"] = ab.score_variants(variants)

    # single-component priors, to see what each one is worth alone
    for comp in [c.name for c in full.components]:
        w = {k: (1.0 if k == comp else 0.0) for k in full.weights}
        only = evidence_mod.load(args.structure_evidence, args.mpnn, args.evidence_base, weights=w)
        priors[f"only_{comp}"] = only.score_variants(variants)

    # the volume term is off by default; evaluate both parameterisations so the
    # decision to drop it is reproducible rather than asserted
    for label, kwargs in (("with_symmetric_volume", {"volume_asymmetry": (1.0, 1.0)}),
                          ("with_asymmetric_volume", {"volume_asymmetry": (1.0, 0.25)})):
        w = dict(full.weights)
        w["burial_volume"] = 1.0
        vp = evidence_mod.load(args.structure_evidence, args.mpnn, args.evidence_base,
                               weights=w, **kwargs)
        priors[label] = vp.score_variants(variants)

    # reference points
    if Path(args.mpnn).exists():
        table = load_log_odds(args.mpnn)
        priors["reference_mpnn_raw"] = score_variants(variants, table)
        priors["reference_mpnn_zscored"] = score_variants(variants, zscore_by_substitution(table))

    singles = n_mut <= 1
    sm = single_mutant_table(variants[singles], fitness[singles])
    priors["leakage_measured_single_mutants"] = score_variants(variants, sm)

    rng = np.random.default_rng(0)
    priors["reference_random"] = rng.normal(size=len(variants))

    report = {
        "n_variants": int(len(variants)),
        "global_optimum": {"variant": str(variants[int(np.argmax(fitness))]),
                           "fitness": float(fitness.max())},
        "note": ("leakage_measured_single_mutants uses measured values from the same deep "
                 "mutational scan that serves as the oracle; it is an upper bound, not a prior "
                 "the campaign is allowed to use"),
        "priors": {name: evaluate(sc, fitness, n_mut) for name, sc in priors.items()},
        "detail": detail,
        "prior_table": full.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    width = max(len(k) for k in report["priors"])
    print(f"{'prior':{width}}  rho_all  rho_hd3  top5%  best96  enrich")
    for name, m in sorted(report["priors"].items(), key=lambda kv: -kv[1]["spearman_hd3plus"]):
        print(f"{name:{width}}  {m['spearman_all']:+.3f}   {m['spearman_hd3plus']:+.3f}  "
              f"{m['top5pct_precision']:.3f}  {m['best_in_top96']:6.3f}  {m['enrichment_top1pct']:5.2f}")


if __name__ == "__main__":
    main()
