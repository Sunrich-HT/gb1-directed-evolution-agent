#!/usr/bin/env python
"""Baseline fitness prediction and high-value identification.

Answers the first required question of the exam: how well can a model trained on
a realistic number of labels rank unseen variants, and how many of the genuinely
high-fitness variants does its top list recover?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import FLIP_SPLITS, flip_split_indices, load_landscape, make_split
from gb1agent.metrics import regression_metrics, top_percentile_precision, topk_recall
from gb1agent.predictors import DEFAULT_GRID, FitnessModel, ModelSpec, select_model


def evaluate(model, variants_test, y_test) -> dict:
    pred = model.predict(variants_test)
    out = regression_metrics(y_test, pred)
    out["top100_recall"] = topk_recall(y_test, pred, k=100)
    out["top5pct_precision"] = top_percentile_precision(y_test, pred, pct=5.0)
    out["best_in_predicted_top96"] = float(np.sort(y_test[np.argsort(-pred)[:96]])[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/baselines.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    args = ap.parse_args()

    df = load_landscape(args.data)
    variants = df["variant"].to_numpy()
    y = df["fitness"].to_numpy(float)
    results = {"protocol": {}, "d0_models": [], "training_size_curve": [], "flip_splits": []}

    # --- 1. models trained on D0 only (the loop's starting information) -------
    for seed in args.seeds:
        sp = make_split(df, seed=seed)
        results["protocol"] = sp.summary()
        spec, report = select_model(variants[sp.initial], y[sp.initial], seed=seed)
        for row in report:
            row.update(seed=seed, selected=row["spec"] == spec.label())
        results.setdefault("cv_report", []).extend(report)
        for candidate in DEFAULT_GRID:
            model = FitnessModel(candidate, seed).fit(variants[sp.initial], y[sp.initial])
            row = evaluate(model, variants[sp.test], y[sp.test])
            row.update(seed=seed, spec=candidate.label(), n_train=int(sp.initial.size),
                       cv_selected=candidate.label() == spec.label())
            results["d0_models"].append(row)

    # --- 2. how the selected model improves with more labels ------------------
    spec = ModelSpec("ridge", "pairwise2480", {"alpha": 0.1})
    sp = make_split(df, seed=args.seeds[0])
    rng = np.random.default_rng(0)
    for n_train in (192, 384, 768, 1536, 3072, 6144):
        idx = rng.choice(sp.pool, n_train, replace=False)
        model = FitnessModel(spec, 0).fit(variants[idx], y[idx])
        row = evaluate(model, variants[sp.test], y[sp.test])
        row.update(n_train=n_train, spec=spec.label())
        results["training_size_curve"].append(row)

    # --- 3. official FLIP benchmark splits ------------------------------------
    for name in FLIP_SPLITS:
        try:
            train, test = flip_split_indices(df, name)
        except KeyError:
            continue
        if train.size == 0 or test.size == 0:
            continue
        model = FitnessModel(spec, 0).fit(variants[train], y[train])
        row = evaluate(model, variants[test], y[test])
        row.update(split=name, n_train=int(train.size), spec=spec.label())
        results["flip_splits"].append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
