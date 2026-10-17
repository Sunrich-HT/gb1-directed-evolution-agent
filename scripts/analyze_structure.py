#!/usr/bin/env python
"""Does predicted-structure confidence add anything over the sequence model?

Inputs : results/structure/structure_features.json (from run_structure_ablation.py)
Output : results/structure/structure_analysis.json

Protocol
--------
* Base predictor: pairwise-encoded ridge trained on D0 (192 labels), no structure.
* Evaluation set: the folded, fitness-stratified sample.
* Ablation: 5-fold CV inside the evaluation set comparing a model on the base
  prediction alone against the same model plus the Boltz-2 confidence columns.
  Both see identical folds, so the delta is attributable to the structure columns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape, make_split
from gb1agent.metrics import top_percentile_precision
from gb1agent.predictors import FitnessModel, ModelSpec

STRUCT_COLS = ["plddt_mean", "plddt_sites", "ptm"]


def cv_scores(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    pred = np.zeros_like(y)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(X):
        model = Ridge(alpha=1.0).fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return float(spearmanr(pred, y).statistic), float(top_percentile_precision(y, pred, 5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", default="results/structure/structure_features.json")
    ap.add_argument("--meta", default="results/structure/structure_meta.json")
    ap.add_argument("--receipts", default="results/structure/nvidia_receipts.jsonl")
    ap.add_argument("--out", default="results/structure/structure_analysis.json")
    ap.add_argument("--seed", type=int, default=301)
    args = ap.parse_args()

    rows = [r for r in json.loads(Path(args.features).read_text()) if "error" not in r]
    meta = json.loads(Path(args.meta).read_text())
    variants = np.array([r["variant"] for r in rows])
    y = np.log1p(np.array([r["fitness"] for r in rows], float))

    land = load_landscape(args.data)
    split = make_split(land, seed=args.seed)
    all_v = land["variant"].to_numpy()
    all_y = land["fitness"].to_numpy(float)
    base_model = FitnessModel(ModelSpec("ridge", "pairwise2480", {"alpha": 1.0}), args.seed)
    base_model.fit(all_v[split.initial], all_y[split.initial])
    base_pred = base_model.predict(variants)

    columns = {c: np.array([r[c] for r in rows], float) for c in STRUCT_COLS}
    columns["sequence_model"] = base_pred
    rho = {k: float(spearmanr(v, y).statistic) for k, v in columns.items()}

    X_seq = base_pred.reshape(-1, 1)
    X_both = np.column_stack([base_pred] + [columns[c] for c in STRUCT_COLS])
    sp_seq, t5_seq = cv_scores(X_seq, y, args.seed)
    sp_both, t5_both = cv_scores(X_both, y, args.seed)

    receipts = [json.loads(line) for line in Path(args.receipts).read_text().splitlines() if line.strip()]
    out = {
        "n_folded": len(rows),
        "n_requested": meta["n_requested"],
        "n_keys_rotated": meta["n_keys_rotated"],
        "calls_per_key": meta["calls_per_key"],
        "n_calls": len(receipts),
        "median_latency_s": meta["median_latency_s"],
        "service": meta["service"],
        "spearman_vs_fitness": rho,
        "max_abs_rho": float(max(abs(rho[c]) for c in STRUCT_COLS)),
        "ablation": {
            "sequence_only": {"spearman": sp_seq, "top5pct_precision": t5_seq},
            "sequence_plus_structure": {"spearman": sp_both, "top5pct_precision": t5_both},
            "delta_spearman": sp_both - sp_seq,
            "delta_top5pct": t5_both - t5_seq,
            "protocol": "5-fold CV inside the folded sample; identical folds for both models",
        },
        "notes": {
            "plddt_mean": "整链平均置信度；四点突变几乎不改变折叠",
            "plddt_sites": "仅取 39/40/41/54 四个位点的置信度",
            "ptm": "整体拓扑置信度",
            "sequence_model": "仅用 192 条标签训练的成对编码岭回归，作为对照上限",
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_folded", "spearman_vs_fitness", "ablation")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
