#!/usr/bin/env python
"""Fold GB1 variants IN COMPLEX with the IgG1 Fc fragment, via hosted Boltz-2.

The monomer fold carries no information about binding - every variant folds into the
same domain.  The phenotype in this landscape is binding to IgG-Fc, so the structure
that can inform a decision is the COMPLEX.  This script folds variant + Fc chain and
keeps the coordinates, so interface geometry can be measured per variant.

Receipts (status, latency, key index) are appended to the NVIDIA receipts log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.nvidia import NvidiaClient

GB1_DOMAIN = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
POSITIONS = (39, 40, 41, 54)
# IgG1 Fc, chain A of RCSB 1FCC - the partner the measured phenotype is about
FC_CHAIN = "PSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPQVKFNWYVDGVQVHNAKTKPREQQYNSTYRVVSVLTVLHQNWLDGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSL"


def variant_sequence(variant: str) -> str:
    seq = list(GB1_DOMAIN)
    for pos, aa in zip(POSITIONS, variant):
        seq[pos - 1] = aa
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", default="results/structure/complex")
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--sampling", type=int, default=50)
    ap.add_argument("--dimer", action="store_true",
                    help="give Boltz-2 the Fc HOMODIMER (the native state) instead of one chain")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = NvidiaClient(receipts_path=Path("results/structure/nvidia_receipts.jsonl"))
    print(f"keys: {client.n_keys} | Fc partner: {len(FC_CHAIN)} aa", flush=True)

    meta_path = out / "complex_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for v in args.variants:
        if (out / f"{v}.cif").exists():
            print(v, "already folded, skipping", flush=True)
            continue
        t0 = time.time()
        payload = [
            {"id": "A", "molecule_type": "protein", "sequence": variant_sequence(v)},
            {"id": "B", "molecule_type": "protein", "sequence": FC_CHAIN},
        ]
        if args.dimer:
            payload.append({"id": "C", "molecule_type": "protein", "sequence": FC_CHAIN})
        resp = client.boltz2(payload, recycling_steps=args.recycling,
                             sampling_steps=args.sampling)
        cif = resp["structures"][0]["structure"]
        (out / f"{v}.cif").write_text(cif)
        meta[v] = {"variant_sequence": variant_sequence(v),
                   "n_chains": len(payload),
                   "n_residues_total": sum(len(p["sequence"]) for p in payload),
                   "confidence_scores": resp.get("confidence_scores"),
                   "cif_bytes": len(cif), "seconds": round(time.time() - t0, 1)}
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{v} ok {len(cif)} bytes {meta[v]['seconds']}s "
              f"conf={meta[v]['confidence_scores']}", flush=True)

    print("done", len(meta), "complexes")


if __name__ == "__main__":
    main()
