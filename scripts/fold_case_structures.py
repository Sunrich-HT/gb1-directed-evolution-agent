#!/usr/bin/env python
"""Fold a handful of named variants with hosted Boltz-2 and KEEP the coordinates.

The ablation run (run_structure_ablation.py) stored only confidence metrics; the
appendix figure needs the actual predicted structures, so this re-requests a few
variants and writes their mmCIF to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.nvidia import NvidiaClient

GB1_DOMAIN = ("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE")
POSITIONS = (39, 40, 41, 54)


def variant_sequence(variant: str) -> str:
    seq = list(GB1_DOMAIN)
    for pos, aa in zip(POSITIONS, variant):
        seq[pos - 1] = aa
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", default="results/structure/cif")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = NvidiaClient(receipts_path=Path("results/structure/nvidia_receipts.jsonl"))
    print(f"keys configured: {client.n_keys}", flush=True)

    meta = {}
    for v in args.variants:
        seq = variant_sequence(v)
        resp = client.boltz2([{"id": "A", "molecule_type": "protein", "sequence": seq}])
        cif = resp["structures"][0]["structure"]
        (out / f"{v}.cif").write_text(cif)
        # the service returns confidence_scores as a list of floats, not JSON blobs
        meta[v] = {"sequence": seq, "cif_bytes": len(cif),
                   "confidence_scores": resp.get("confidence_scores"),
                   "metrics": {k: resp[k] for k in ("complex_plddt", "ptm", "confidence_score")
                               if k in resp}}
        print(v, "ok", len(cif), "bytes", flush=True)

    (out / "fold_meta.json").write_text(json.dumps(meta, indent=2))
    print("done")


if __name__ == "__main__":
    main()
