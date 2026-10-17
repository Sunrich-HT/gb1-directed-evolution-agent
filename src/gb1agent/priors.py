"""Zero-shot priors for round 1, before any multi-mutant label exists.

Two sources, both usable with zero measured multi-mutant data:

* ``esm2_log_odds``  -- masked-marginal log-odds from ESM-2 at the four mutable
  positions.  Because the four sites are scored independently the whole prior is
  an 80-number table (4 sites x 20 amino acids), computed once on a GPU and
  cached in ``config/esm2_log_odds.json`` so the repo reproduces offline.
* ``single_mutant``  -- additive extrapolation from the published single-mutant
  measurements (76 values, Wu et al. 2016).

``zscore_by_substitution`` implements the normalisation of Tran et al. (Science
2026): protein language models systematically penalise proline and bulky
aromatics, so raw log-odds under-rank exactly the substitutions that turn out to
be function-enhancing.  Z-scoring within each target amino acid removes that
bias.  On GB1 this matters: two of the four components of the global optimum
FWAA (V39F and G41A) are deleterious as single mutants and are discarded by any
"keep the enhancing singles" filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import AA, AA_INDEX

WT = "VDGV"
POSITIONS = (39, 40, 41, 54)


def load_log_odds(path: str | Path) -> np.ndarray:
    """Load the cached 4x20 ESM-2 masked-marginal table."""
    table = json.loads(Path(path).read_text())
    arr = np.zeros((4, 20), float)
    for site, row in enumerate(table["log_odds"]):
        for aa, value in row.items():
            arr[site, AA_INDEX[aa]] = value
    return arr


def zscore_by_substitution(log_odds: np.ndarray) -> np.ndarray:
    """Normalise each target amino acid across sites (Tran et al. 2026)."""
    mu = log_odds.mean(axis=0, keepdims=True)
    sd = log_odds.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (log_odds - mu) / sd


def score_variants(variants, table: np.ndarray) -> np.ndarray:
    """Sum the per-site contributions of a 4x20 table over each variant."""
    out = np.zeros(len(variants), float)
    for r, v in enumerate(variants):
        out[r] = sum(table[i, AA_INDEX[a]] for i, a in enumerate(v))
    return out


def single_mutant_table(variants, fitness) -> np.ndarray:
    """4x20 table of log-fitness effects measured for single mutants."""
    table = np.full((4, 20), np.nan)
    lookup = dict(zip(variants, np.log1p(np.clip(np.asarray(fitness, float), 0, None))))
    wt_value = lookup.get(WT, 0.0)
    for v, value in lookup.items():
        diff = [i for i in range(4) if v[i] != WT[i]]
        if len(diff) == 1:
            table[diff[0], AA_INDEX[v[diff[0]]]] = value - wt_value
    for i in range(4):
        table[i, AA_INDEX[WT[i]]] = 0.0
    # unmeasured substitutions: pessimistic but finite
    filled = np.nanmin(table) if np.isfinite(np.nanmin(table)) else -2.0
    return np.where(np.isnan(table), filled, table)


def diversified_topk(scores: np.ndarray, k: int, pool_top: int, rng) -> np.ndarray:
    """Sample k candidates from the prior's top ``pool_top``.

    Taking the prior's literal top-k collapses batch diversity -- on GB1 the top
    single-mutant substitutions are almost all at position 40 -- which starves
    the first model of contrast.  Sampling from a wider prior shortlist keeps the
    head start without the collapse.
    """
    order = np.argsort(-scores)[:pool_top]
    return rng.choice(order, size=min(k, order.size), replace=False)
