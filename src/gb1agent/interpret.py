"""Interpretability: what the model learned, what the agent decided, and why.

Three layers, each answering a question a reviewer will actually ask.

1. Model layer   -- which substitutions carry the signal, and where does the
                    pairwise model place epistasis?
2. Decision layer-- what did the agent claim, did the data support it, and did
                    the next round change as a result?
3. Failure layer -- which structural property of this landscape defeats the
                    obvious strategies?
"""
from __future__ import annotations

import numpy as np

from .features import AA, AA_INDEX, PAIRS
from .predictors import FitnessModel, ModelSpec, to_target

POSITIONS = (39, 40, 41, 54)
WT = "VDGV"


def main_effect_matrix(model: FitnessModel) -> np.ndarray:
    """4 x 20 matrix of additive coefficients from a fitted linear model."""
    coef = np.asarray(model._model.coef_, float).ravel()
    return coef[:80].reshape(4, 20)


def epistasis_matrix(model: FitnessModel) -> np.ndarray:
    """6 x 20 x 20 interaction coefficients for the six site pairs."""
    coef = np.asarray(model._model.coef_, float).ravel()
    if coef.size < 2480:
        raise ValueError("model has no pairwise block")
    return coef[80:2480].reshape(6, 20, 20)


def epistasis_strength(model: FitnessModel) -> dict:
    """How much weight the model puts on each site pair, relative to additive."""
    main = np.abs(main_effect_matrix(model)).sum()
    inter = epistasis_matrix(model)
    per_pair = {
        f"{POSITIONS[i]}-{POSITIONS[j]}": float(np.abs(inter[p]).sum())
        for p, (i, j) in enumerate(PAIRS)
    }
    total = sum(per_pair.values())
    return {
        "additive_l1": float(main),
        "pairwise_l1": float(total),
        "pairwise_share": float(total / (total + main)) if (total + main) else 0.0,
        "per_pair_l1": per_pair,
    }


def measured_main_effects(variants, fitness) -> np.ndarray:
    """4 x 20 matrix of measured single-mutant log-fitness effects (NaN if absent)."""
    table = np.full((4, 20), np.nan)
    lookup = dict(zip(variants, to_target(fitness)))
    wt = lookup.get(WT, 0.0)
    for v, value in lookup.items():
        diff = [i for i in range(4) if v[i] != WT[i]]
        if len(diff) == 1:
            table[diff[0], AA_INDEX[v[diff[0]]]] = value - wt
    return table


def deleterious_components_of(variant: str, variants, fitness) -> list[dict]:
    """Single-mutant values of a target variant's components.

    On GB1 this is the crux of the whole problem: the global optimum FWAA is
    built from two substitutions that are harmful on their own, so any strategy
    that only combines individually-enhancing mutations cannot reach it.
    """
    lookup = dict(zip(variants, np.asarray(fitness, float)))
    rows = []
    singles = {
        v: y for v, y in lookup.items()
        if sum(a != b for a, b in zip(v, WT)) == 1
    }
    ranked = sorted(singles.items(), key=lambda kv: -kv[1])
    rank_of = {v: i + 1 for i, (v, _) in enumerate(ranked)}
    for i, aa in enumerate(variant):
        if aa == WT[i]:
            continue
        single = list(WT)
        single[i] = aa
        key = "".join(single)
        rows.append({
            "mutation": f"{WT[i]}{POSITIONS[i]}{aa}",
            "single_mutant_fitness": float(lookup.get(key, float("nan"))),
            "rank_among_singles": int(rank_of.get(key, -1)),
            "deleterious_alone": bool(lookup.get(key, 0.0) < 1.0),
        })
    return rows


def hypothesis_trajectory(records) -> list[dict]:
    """Flatten every registered hypothesis with its program-assigned verdict."""
    rows = []
    for r in records:
        for h in r.get("hypotheses", []):
            rows.append({
                "arm": r["arm"], "seed": r["seed"], "round": r["round"],
                "statement": h["statement"], "site": h["site"],
                "substitution": h["substitution"], "direction": h["direction"],
                "verdict": h["verdict"], **{f"ev_{k}": v for k, v in h["evidence"].items()},
            })
    return rows


def exploration_profile(records, predictions_key: str = "predicted") -> list[dict]:
    """Per round: how far the batch sits from the model's own top ranking."""
    rows = []
    for r in records:
        preds = np.array([s[predictions_key] for s in r["selected"]], float)
        obs = np.array([s["fitness"] for s in r["selected"]], float)
        rows.append({
            "arm": r["arm"], "seed": r["seed"], "round": r["round"],
            "pred_mean": float(preds.mean()), "pred_min": float(preds.min()),
            "observed_mean": float(obs.mean()), "observed_best": float(obs.max()),
            "calibration_bias": float((obs - preds).mean()),
            "mean_pairwise_hamming": r["diversity"]["mean_pairwise_hamming"],
            "unique_aa_per_site": r["diversity"]["unique_aa_per_site"],
        })
    return rows
