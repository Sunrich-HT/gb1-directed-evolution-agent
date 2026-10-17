#!/usr/bin/env python
"""Two analyses the task asks for explicitly.

1. mutation order - do single, double, triple and quadruple mutants behave
   differently, and does the predictor handle them equally well?  This is where
   epistasis shows up as a prediction problem rather than as a landscape property.

2. uncertainty - an ensemble of predictors trained on bootstrap resamples of the
   same labels, so every candidate carries a spread as well as a mean.  The question
   is whether that spread is informative: are the model's confident predictions
   actually more accurate, and would an uncertainty-aware pick beat a plain greedy one?

Writes results/mutation_order_and_uncertainty.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape, make_split
from gb1agent.metrics import regression_metrics, top_percentile_precision, topk_recall
from gb1agent.predictors import FitnessModel, select_model

WT = "VDGV"


def n_mut(v: str) -> int:
    return sum(1 for a, b in zip(v, WT) if a != b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--n-ensemble", type=int, default=12)
    ap.add_argument("--out", default="results/mutation_order_and_uncertainty.json")
    args = ap.parse_args()

    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)
    orders = np.array([n_mut(str(v)) for v in variants])

    split = make_split(land, seed=args.seed)
    known = np.array(split.initial)
    test = np.array(split.test)

    out = {"seed": args.seed}

    # ---------------------------------------------------------------- 1. order
    by_order = {}
    for k in (1, 2, 3, 4):
        m = orders == k
        by_order[str(k)] = {
            "n_variants": int(m.sum()),
            "mean_fitness": float(fitness[m].mean()),
            "median_fitness": float(np.median(fitness[m])),
            "best_fitness": float(fitness[m].max()),
            "frac_above_wild_type": float((fitness[m] > 1.0).mean()),
            "best_variant": str(variants[m][int(np.argmax(fitness[m]))]),
        }
    out["landscape_by_mutation_order"] = by_order

    chosen = select_model(variants[known], fitness[known], seed=args.seed)
    spec = chosen[0] if isinstance(chosen, tuple) else chosen
    model = FitnessModel(spec, args.seed).fit(variants[known], fitness[known])
    pred_test = model.predict(variants[test])
    acc = {}
    for k in (1, 2, 3, 4):
        m = orders[test] == k
        if m.sum() < 20:
            continue
        acc[str(k)] = {"n_test": int(m.sum()),
                       **regression_metrics(fitness[test][m], pred_test[m])}
    out["predictor_accuracy_by_order"] = acc
    out["predictor_spec"] = f"{spec.kind}[{spec.encoding}] {spec.params}"

    # ---------------------------------------------------------------- 2. uncertainty
    rng = np.random.default_rng(args.seed)
    members = []
    for b in range(args.n_ensemble):
        idx = rng.choice(len(known), size=len(known), replace=True)
        mb = FitnessModel(spec, args.seed + b).fit(variants[known][idx], fitness[known][idx])
        members.append(mb.predict(variants[test]))
    P = np.vstack(members)
    mean, sd = P.mean(0), P.std(0)

    # is the spread informative?  split the test set by predicted uncertainty
    lo = sd <= np.median(sd)
    ens = {
        "n_members": args.n_ensemble,
        "ensemble_mean": regression_metrics(fitness[test], mean),
        "single_model": regression_metrics(fitness[test], pred_test),
        "accuracy_on_confident_half": regression_metrics(fitness[test][lo], mean[lo]),
        "accuracy_on_uncertain_half": regression_metrics(fitness[test][~lo], mean[~lo]),
        "spearman_sd_vs_abs_error": float(spearmanr(
            sd, np.abs(mean - fitness[test])).statistic),
    }

    # would an uncertainty-aware pick beat a plain greedy pick, at batch size 96?
    picks = {}
    for label, score in (("greedy_mean", mean),
                         ("ucb_mean_plus_1sd", mean + sd),
                         ("ucb_mean_plus_2sd", mean + 2 * sd),
                         ("pure_exploration_sd", sd)):
        sel = np.argsort(-score)[:96]
        picks[label] = {"best": float(fitness[test][sel].max()),
                        "mean": float(fitness[test][sel].mean()),
                        "top5pct_precision": top_percentile_precision(
                            fitness[test], score, 5.0),
                        "top100_recall": topk_recall(fitness[test], score, 100)}
    ens["batch_of_96_from_the_test_pool"] = picks
    out["uncertainty"] = ens

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"by_order": {k: round(v["mean_fitness"], 3) for k, v in by_order.items()},
                      "acc_spearman": {k: round(v["spearman"], 3) for k, v in acc.items()},
                      "sd_vs_error_rho": round(ens["spearman_sd_vs_abs_error"], 3),
                      "picks": {k: round(v["best"], 3) for k, v in picks.items()}},
                     ensure_ascii=False, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
