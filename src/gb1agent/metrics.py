"""Evaluation metrics for fitness prediction and for the evolution campaign."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rho = spearmanr(y_true, y_pred).statistic
    r = pearsonr(y_true, y_pred).statistic
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "n": int(y_true.size),
        "spearman": float(rho),
        "pearson": float(r),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def topk_recall(y_true, y_pred, k: int = 100) -> float:
    """Fraction of the true top-k that appears in the predicted top-k."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    return len(true_top & pred_top) / k


def top_percentile_precision(y_true, y_pred, pct: float = 5.0) -> float:
    """MULTI-evolve's precision metric: of the predicted top pct%, what fraction
    is truly in the top pct%?  (Tran et al., Science 2026.)"""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    k = max(1, int(round(y_true.size * pct / 100.0)))
    threshold = np.quantile(y_true, 1 - pct / 100.0)
    picked = np.argsort(-y_pred)[:k]
    return float((y_true[picked] >= threshold).mean())


def campaign_metrics(selected_fitness, incumbent_before: float) -> dict:
    """Per-round summary of a directed-evolution batch."""
    xs = np.asarray(selected_fitness, float)
    best = float(xs.max())
    return {
        "n_selected": int(xs.size),
        "mean": float(xs.mean()),
        "best": best,
        "incumbent_before": float(incumbent_before),
        "gain": float(max(0.0, best - incumbent_before)),
        "hit_rate_above_wt": float((xs > 1.0).mean()),
    }


def batch_diversity(variants) -> dict:
    """How spread out a selected batch is -- the quantity a sharp prior destroys."""
    variants = list(variants)
    n = len(variants)
    per_site = []
    for i in range(4):
        per_site.append(len({v[i] for v in variants}))
    dists = [
        sum(a != b for a, b in zip(variants[i], variants[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return {
        "unique_aa_per_site": per_site,
        "mean_pairwise_hamming": float(np.mean(dists)) if dists else 0.0,
    }
