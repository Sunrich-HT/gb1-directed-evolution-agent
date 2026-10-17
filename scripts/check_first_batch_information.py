#!/usr/bin/env python
"""How much does each arm's FIRST batch teach the round-2 model?

Post-hoc check behind appendix A.3.  For one seed, take the same 192 starting
labels, append each arm's round-1 batch of 96, refit a predictor with the same
cross-validated procedure the campaign uses, and ask where that model ranks the
global optimum in the untouched candidate pool.  Nothing here feeds back into any
campaign; it only explains a result that already happened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape, make_split
from gb1agent.predictors import FitnessModel, select_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--arms", nargs="+",
                    default=["prior_constraint", "prior_fixed", "greedy"])
    ap.add_argument("--out", default="results/case_round2_model_check.json")
    args = ap.parse_args()

    df = load_landscape(args.data)
    variants = df["variant"].to_numpy()
    fitness = df["fitness"].to_numpy(float)
    index = {v: i for i, v in enumerate(variants)}
    target = int(np.argmax(fitness))

    split = make_split(df, seed=args.seed)
    records = json.loads(Path(args.records).read_text())

    out = []
    for arm in args.arms:
        r1 = [r for r in records
              if r["arm"] == arm and r["seed"] == args.seed and r["round"] == 1][0]
        known = list(split.initial) + [index[d["variant"]] for d in r1["selected"]]
        chosen = select_model(variants[known], fitness[known], seed=args.seed)
        spec = chosen[0] if isinstance(chosen, tuple) else chosen
        model = FitnessModel(spec, args.seed).fit(variants[known], fitness[known])

        pool = np.array([i for i in split.pool if i not in set(known)])
        pred = model.predict(variants[pool])
        here = int(np.where(pool == target)[0][0])
        rank = int((pred > pred[here]).sum()) + 1
        top96 = pool[np.argsort(-pred)[:96]]

        out.append({
            "arm": arm, "seed": args.seed, "n_labels": len(known),
            "spec": f"{spec.kind}[{spec.encoding}]",
            "best_label_r1": float(fitness[known].max()),
            "target": str(variants[target]),
            "fwaa_pred": float(pred[here]), "fwaa_rank": rank, "n_pool": int(len(pool)),
            "in_top96": bool(rank <= 96),
            "best_in_model_top96": float(fitness[top96].max()),
        })
        print(f"{arm:<17} rank {rank:>7,} / {len(pool):,} | "
              f"best inside model top96 {fitness[top96].max():.3f}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
