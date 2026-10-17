"""Fitness predictors and model selection by cross-validation on D only."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from .features import ENCODERS
from .metrics import regression_metrics

# Training target: log1p(fitness).  The raw selection ratio spans four orders of
# magnitude with a large zero mass, so a linear model fitted on raw values is
# dominated by a handful of high points.
def to_target(y):
    return np.log1p(np.clip(np.asarray(y, float), 0.0, None))


def from_target(z):
    return np.expm1(np.asarray(z, float))


@dataclass
class ModelSpec:
    kind: str = "ridge"
    encoding: str = "pairwise2480"
    params: dict = field(default_factory=dict)

    def build(self, seed: int = 0):
        if self.kind == "ridge":
            return Ridge(alpha=self.params.get("alpha", 1.0), solver="lsqr", tol=1e-8)
        if self.kind == "rf":
            return RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 300),
                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                max_features=self.params.get("max_features", "sqrt"),
                random_state=seed,
                n_jobs=-1,
            )
        if self.kind == "fcnn":
            return MLPRegressor(
                hidden_layer_sizes=self.params.get("hidden", (256, 64)),
                max_iter=self.params.get("max_iter", 2000),
                random_state=seed,
            )
        raise ValueError(f"unknown model kind {self.kind}")

    def label(self) -> str:
        extra = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}[{self.encoding}]" + (f"({extra})" if extra else "")


class FitnessModel:
    """Encoder + estimator, fitted on labelled variants only."""

    def __init__(self, spec: ModelSpec, seed: int = 0):
        self.spec = spec
        self.seed = seed
        self._model = spec.build(seed)
        self._dense = spec.kind != "ridge"

    def _encode(self, variants):
        X = ENCODERS[self.spec.encoding](list(variants))
        return X.toarray() if self._dense else X

    def fit(self, variants, fitness):
        self._model.fit(self._encode(variants), to_target(fitness))
        return self

    def predict(self, variants) -> np.ndarray:
        """Predicted fitness on the original scale."""
        return from_target(self._model.predict(self._encode(variants)))


DEFAULT_GRID = [
    ModelSpec("ridge", "additive80", {"alpha": 1.0}),
    ModelSpec("ridge", "pairwise2480", {"alpha": 0.1}),
    ModelSpec("ridge", "pairwise2480", {"alpha": 1.0}),
    ModelSpec("ridge", "pairwise2480", {"alpha": 10.0}),
    ModelSpec("rf", "additive80", {"min_samples_leaf": 1}),
    ModelSpec("fcnn", "pairwise2480", {}),
]


def select_model(variants, fitness, seed: int = 0, grid=None, n_splits: int = 3):
    """Pick a spec by cross-validated Spearman using only the labelled set.

    No test-set or pool label is touched here -- that is what makes the final
    test number an honest estimate.
    """
    grid = list(grid or DEFAULT_GRID)
    variants = list(variants)
    y = np.asarray(fitness, float)
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(variants))
    report = []
    for order, spec in enumerate(grid):
        scores = []
        for train_idx, val_idx in folds:
            model = FitnessModel(spec, seed).fit(
                [variants[i] for i in train_idx], y[train_idx]
            )
            pred = model.predict([variants[i] for i in val_idx])
            scores.append(regression_metrics(y[val_idx], pred))
        report.append(
            {
                "order": order,
                "spec": spec.label(),
                "cv_spearman": float(np.mean([s["spearman"] for s in scores])),
                "cv_rmse": float(np.mean([s["rmse"] for s in scores])),
            }
        )
    best = max(report, key=lambda r: (r["cv_spearman"], -r["cv_rmse"], -r["order"]))
    return grid[best["order"]], report
