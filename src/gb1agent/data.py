"""GB1 four-site landscape: loading, fixed splits and the virtual oracle.

Data source (pinned): FLIP, commit 62cace8735f5610e2743cf06ce0f944b37fffaa6,
splits/gb1/four_mutations_full_data.csv.zip
  zip  SHA256 85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff

The measured fitness column is used as a *virtual* evaluator: no new wet-lab
measurement is claimed anywhere in this project. A label is only revealed to an
agent after that agent has committed to selecting the variant.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WT = "VDGV"
POSITIONS = (39, 40, 41, 54)
ZIP_SHA256 = "85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"
DATA_URL = (
    "https://raw.githubusercontent.com/J-SNACKKB/FLIP/"
    "62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/four_mutations_full_data.csv.zip"
)


def load_landscape(path: str | Path, verify: bool = True) -> pd.DataFrame:
    """Load the full four-site landscape from the pinned FLIP zip."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify and digest != ZIP_SHA256:
        raise ValueError(f"unexpected data checksum {digest}; expected {ZIP_SHA256}")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, low_memory=False)
    df = df.rename(columns={"Variants": "variant", "Fitness": "fitness", "HD": "n_mut"})
    keep = ["variant", "fitness", "n_mut", "sequence"]
    keep += [c for c in ("one_vs_rest", "two_vs_rest", "three_vs_rest", "sampled", "low_vs_high") if c in df.columns]
    df = df[keep].reset_index(drop=True)
    if len(df) != 149361 or float(df.loc[df.variant == WT, "fitness"].iloc[0]) != 1.0:
        raise ValueError("landscape failed sanity check (size or WT fitness)")
    return df


@dataclass(frozen=True)
class Split:
    """A frozen train/candidate/test partition of the landscape.

    test      held out for predictor evaluation; never selectable by any agent
    initial   D0, the labels every arm starts from (identical across arms)
    pool      U, the legal candidate pool the agents select from
    """

    test: np.ndarray
    initial: np.ndarray
    pool: np.ndarray
    seed: int
    n_total: int

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "n_total": self.n_total,
            "n_test": int(self.test.size),
            "n_initial": int(self.initial.size),
            "n_pool": int(self.pool.size),
            "disjoint": True,
        }


def make_split(df: pd.DataFrame, seed: int, n_test: int = 10000, n_initial: int = 192) -> Split:
    """Fixed, documented partition.

    The test set is drawn with a *protocol* seed shared by every experiment so
    that predictor numbers are comparable across arms and runs; D0 is drawn with
    the per-experiment seed.  Crucially the candidate pool keeps the whole
    remaining landscape -- including its global optimum -- so that "did the loop
    find anything better than it started with" is an answerable question.
    """
    n = len(df)
    protocol_rng = np.random.default_rng(20261017)
    perm = protocol_rng.permutation(n)
    test = np.sort(perm[:n_test])
    rest = perm[n_test:]
    rng = np.random.default_rng(seed)
    initial = np.sort(rng.choice(rest, n_initial, replace=False))
    pool = np.sort(np.setdiff1d(rest, initial, assume_unique=False))
    return Split(test=test, initial=initial, pool=pool, seed=seed, n_total=n)


class Oracle:
    """Reveals a measured label only for variants an agent has committed to."""

    def __init__(self, df: pd.DataFrame, log_path: str | Path | None = None):
        self._y = df["fitness"].to_numpy(float)
        self._queries: list[dict] = []
        self._log_path = Path(log_path) if log_path else None

    @property
    def n_queries(self) -> int:
        return sum(q["n"] for q in self._queries)

    def reveal(self, indices, *, arm: str, round_: int, commit_sha256: str) -> np.ndarray:
        indices = np.asarray(indices, int)
        if indices.size != np.unique(indices).size:
            raise ValueError("duplicate indices in a single commitment")
        self._queries.append(
            {"arm": arm, "round": round_, "n": int(indices.size), "commit_sha256": commit_sha256}
        )
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as fh:
                fh.write(json.dumps(self._queries[-1]) + "\n")
        return self._y[indices]

    def truth_for_evaluation(self, indices) -> np.ndarray:
        """Labels for the frozen test set -- reporting only, never fed to an agent."""
        return self._y[np.asarray(indices, int)]


def commit_digest(variants) -> str:
    payload = json.dumps(sorted(map(str, variants)), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


FLIP_SPLITS = ("sampled", "two_vs_rest", "three_vs_rest", "low_vs_high")


def flip_split_indices(df: pd.DataFrame, name: str):
    """Train/test indices of an official FLIP benchmark split."""
    if name not in df.columns:
        raise KeyError(f"split column {name} not present")
    col = df[name]
    train = np.where(col == "train")[0]
    test = np.where(col == "test")[0]
    return train, test
