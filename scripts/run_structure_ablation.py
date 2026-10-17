#!/usr/bin/env python
"""Structure track: fold GB1 variants with NVIDIA-hosted Boltz-2 and ask whether
predicted-structure confidence carries information the sequence models lack.

Design
------
* A fitness-stratified sample of measured variants is folded once each.
* Features taken from the prediction: global complex pLDDT, mean per-residue
  pLDDT over the four mutable sites, and pTM.
* Requests rotate over the keys in ``NVIDIA_API_KEYS`` (comma separated); a key
  is never logged -- receipts record the key *index*, HTTP status and latency.

Run:  python scripts/run_structure_ablation.py --data data/gb1_flip.zip --n 150
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape
from gb1agent.nvidia import NvidiaClient

GB1_DOMAIN = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POSITIONS = (39, 40, 41, 54)


def variant_sequence(variant: str) -> str:
    seq = list(GB1_DOMAIN)
    for pos, aa in zip(POSITIONS, variant):
        seq[pos - 1] = aa
    return "".join(seq)


def per_residue_plddt(cif_text: str) -> np.ndarray:
    """Mean per-residue pLDDT from the mmCIF ``atom_site`` loop.

    The service sometimes returns rows broken across lines, so the loop body is
    tokenised as a flat stream and chunked by the number of declared columns
    rather than parsed line by line.
    """
    lines = cif_text.splitlines()
    columns: list[str] = []
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1])
            start = i + 1
        elif columns and start is not None and i >= start:
            break
    if not columns:
        raise ValueError("no _atom_site loop in the returned mmCIF")
    body: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped in ("#", "") or stripped.startswith(("loop_", "data_", "_")):
            if body:
                break
            continue
        body.extend(stripped.split())
    if len(body) % len(columns):
        body = body[: len(body) - len(body) % len(columns)]
    table = np.array(body, dtype=object).reshape(-1, len(columns))
    seq_col = columns.index("label_seq_id")
    b_col = columns.index("B_iso_or_equiv")
    values: dict[int, list[float]] = {}
    for row in table:
        if row[0] != "ATOM":
            continue
        values.setdefault(int(row[seq_col]), []).append(float(row[b_col]))
    if not values:
        raise ValueError("no parsable ATOM records in the returned mmCIF")
    return np.array([np.mean(values[i]) for i in sorted(values)])


def stratified_sample(fitness: np.ndarray, n: int, seed: int, n_bins: int = 10) -> np.ndarray:
    """Equal-count sample from equal-width bins of log1p(fitness).

    Rank deciles would put 9 of 10 bins in the dead-variant mass (77 % of this
    landscape sits below fitness 0.01), leaving the analysis with no dynamic
    range; binning on the value covers the whole fitness axis instead.
    """
    rng = np.random.default_rng(seed)
    logf = np.log1p(fitness)
    edges = np.linspace(logf.min(), logf.max(), n_bins + 1)
    per = max(1, n // n_bins)
    picked = []
    for i in range(n_bins):
        hi = logf <= edges[i + 1] if i == n_bins - 1 else logf < edges[i + 1]
        members = np.flatnonzero((logf >= edges[i]) & hi)
        if len(members):
            picked.append(rng.choice(members, size=min(per, len(members)), replace=False))
    return np.concatenate(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/structure")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=907)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    land = load_landscape(args.data)
    idx = stratified_sample(land.fitness.values, args.n, args.seed)
    variants = land.variant.values[idx]
    fitness = land.fitness.values[idx]
    n_mut = land.n_mut.values[idx]

    client = NvidiaClient(receipts_path=out / "nvidia_receipts.jsonl")
    print(f"folding {len(variants)} variants with {client.n_keys} rotating keys", flush=True)

    lock = threading.Lock()
    rows: list[dict] = []

    def fold(i: int) -> None:
        variant = str(variants[i])
        try:
            payload = client.boltz2(
                [{"id": "A", "molecule_type": "protein", "sequence": variant_sequence(variant)}],
                recycling_steps=args.recycling_steps, sampling_steps=args.sampling_steps,
            )
            cif = payload["structures"][0]["structure"]
            plddt = per_residue_plddt(cif)
        except Exception as exc:  # recorded, not retried silently
            with lock:
                rows.append({"variant": variant, "fitness": float(fitness[i]),
                             "n_mut": int(n_mut[i]), "error": f"{type(exc).__name__}: {exc}"[:120]})
            return
        row = {
            "variant": variant,
            "fitness": float(fitness[i]),
            "n_mut": int(n_mut[i]),
            "plddt_mean": float(plddt.mean()),
            "plddt_sites": float(np.mean([plddt[p - 1] for p in POSITIONS])),
            "plddt_min": float(plddt.min()),
            "complex_plddt": float(np.mean(payload.get("complex_plddt_scores") or [np.nan])),
            "ptm": float(np.mean(payload.get("ptm_scores") or [np.nan])),
            "confidence": float(np.mean(payload.get("confidence_scores") or [np.nan])),
        }
        with lock:
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"  {len(rows)}/{len(variants)} folded", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fold, range(len(variants))))

    rows.sort(key=lambda r: r["variant"])
    (out / "structure_features.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if "error" not in r]
    calls = client.records
    meta = {
        "service": "NVIDIA hosted Boltz-2 (health.api.nvidia.com/v1/biology/mit/boltz2/predict)",
        "n_requested": len(variants), "n_folded": len(ok),
        "n_keys_rotated": client.n_keys,
        "calls_per_key": {str(i): sum(1 for c in calls if c.key_index == i) for i in range(client.n_keys)},
        "http_status_counts": {str(s): sum(1 for c in calls if c.status == s) for s in sorted({c.status for c in calls})},
        "median_latency_s": float(np.median([c.seconds for c in calls])) if calls else None,
        "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
        "sample_seed": args.seed, "sampling": "equal count per equal-width log1p(fitness) bin of the full landscape",
    }
    (out / "structure_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
