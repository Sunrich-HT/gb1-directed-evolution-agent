"""Sequence encodings for the four mutable GB1 sites."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import csr_matrix, hstack

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
N_SITES = 4
PAIRS = list(itertools.combinations(range(N_SITES), 2))


def onehot(variants) -> csr_matrix:
    """80-dim additive encoding (4 sites x 20 amino acids)."""
    rows, cols = [], []
    for r, v in enumerate(variants):
        for i, a in enumerate(v):
            rows.append(r)
            cols.append(20 * i + AA_INDEX[a])
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(len(variants), 80))


def pairwise(variants) -> csr_matrix:
    """2480-dim encoding: 80 additive + 6 site-pairs x 400 interaction terms.

    The interaction block is what lets a linear model represent epistasis, the
    property that makes this landscape hard (Wu et al. 2016).
    """
    rows, cols = [], []
    for r, v in enumerate(variants):
        codes = [AA_INDEX[a] for a in v]
        for i, c in enumerate(codes):
            rows.append(r)
            cols.append(20 * i + c)
        for p, (i, j) in enumerate(PAIRS):
            rows.append(r)
            cols.append(80 + 400 * p + 20 * codes[i] + codes[j])
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(len(variants), 2480))


def with_extra(base: csr_matrix, extra: np.ndarray) -> csr_matrix:
    """Append dense columns (zero-shot priors, structure features) to an encoding."""
    extra = np.asarray(extra, float)
    if extra.ndim == 1:
        extra = extra[:, None]
    return csr_matrix(hstack([base, csr_matrix(extra)]))


ENCODERS = {"additive80": onehot, "pairwise2480": pairwise}
