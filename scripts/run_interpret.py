#!/usr/bin/env python
"""Interpretability report: model layer, decision layer, failure layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape, make_split
from gb1agent.interpret import (
    deleterious_components_of, epistasis_strength, exploration_profile,
    hypothesis_trajectory, main_effect_matrix, measured_main_effects,
)
from gb1agent.predictors import FitnessModel, ModelSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", default="results/experiment/records.json")
    ap.add_argument("--out", default="results/interpretability.json")
    ap.add_argument("--seed", type=int, default=301)
    args = ap.parse_args()

    df = load_landscape(args.data)
    variants = df["variant"].to_numpy()
    fitness = df["fitness"].to_numpy(float)
    split = make_split(df, seed=args.seed)

    spec = ModelSpec("ridge", "pairwise2480", {"alpha": 1.0})
    model_d0 = FitnessModel(spec, args.seed).fit(variants[split.initial], fitness[split.initial])

    report = {
        "model_layer": {
            "epistasis_d0": epistasis_strength(model_d0),
            "main_effects_d0": main_effect_matrix(model_d0).tolist(),
            "measured_single_mutant_effects": np.nan_to_num(
                measured_main_effects(variants, fitness), nan=0.0).tolist(),
        },
        "failure_layer": {
            "global_optimum": str(variants[int(np.argmax(fitness))]),
            "global_optimum_fitness": float(fitness.max()),
            "components": deleterious_components_of(
                str(variants[int(np.argmax(fitness))]), variants, fitness),
        },
    }

    records_path = Path(args.records)
    if records_path.exists():
        records = json.loads(records_path.read_text())
        if records:
            model_final = FitnessModel(spec, args.seed)
            report["decision_layer"] = {
                "hypotheses": hypothesis_trajectory(records),
                "exploration": exploration_profile(records),
            }
            verdicts: dict[str, int] = {}
            for row in report["decision_layer"]["hypotheses"]:
                verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
            report["decision_layer"]["verdict_counts"] = verdicts

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"global_optimum": report["failure_layer"]["global_optimum"],
                      "components": report["failure_layer"]["components"],
                      "epistasis_share": report["model_layer"]["epistasis_d0"]["pairwise_share"]},
                     indent=2))


if __name__ == "__main__":
    main()
