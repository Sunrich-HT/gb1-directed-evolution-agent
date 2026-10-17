#!/usr/bin/env python
"""Fetch the inputs the pipeline needs: the landscape and the reference structures.

Nothing here is bundled with the repository - the dataset carries its own licence and
the structures are better taken from RCSB than from a copy.  Both are verified by
checksum where a checksum is known, and the download receipts are printed so a run can
be audited later.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

FLIP_URL = ("https://raw.githubusercontent.com/J-SNACKKB/FLIP/"
            "62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/"
            "four_mutations_full_data.csv.zip")
FLIP_SHA256 = "85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"
RCSB = "https://files.rcsb.org/download/{}.pdb"
STRUCTURES = {
    "1pga": "free B1 domain, X-ray 2.07 A - the structural layer is computed on this",
    "2gb1": "free B1 domain, NMR - cross-check that disagrees on the loop geometry",
    "1fcc": "protein G C2 domain bound to IgG1 Fc - the interface layer",
}


def fetch(url: str, dest: Path, expect_sha256: str | None = None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as resp:
        body = resp.read()
        status = resp.status
    digest = hashlib.sha256(body).hexdigest()
    if expect_sha256 and digest != expect_sha256:
        sys.exit(f"checksum mismatch for {url}\n  expected {expect_sha256}\n  got      {digest}")
    dest.write_bytes(body)
    return {"url": url, "status": status, "bytes": len(body), "sha256": digest,
            "seconds": round(time.time() - t0, 2), "path": str(dest)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    d = Path(args.data_dir)

    target = d / "gb1_flip.zip"
    if target.exists() and not args.force:
        print(f"{target} already present (use --force to re-download)")
    else:
        r = fetch(FLIP_URL, target, FLIP_SHA256)
        print(f"FLIP GB1  HTTP {r['status']}  {r['bytes']:,} B  sha256 {r['sha256'][:16]}...  "
              f"{r['seconds']}s")

    for pdb, why in STRUCTURES.items():
        out = d / f"{pdb}.pdb"
        if out.exists() and not args.force:
            print(f"{out} already present")
            continue
        r = fetch(RCSB.format(pdb.upper()), out)
        print(f"{pdb.upper():>5}  HTTP {r['status']}  {r['bytes']:,} B  {r['seconds']}s   {why}")

    print("\nProteinMPNN weights are not fetched here: the inverse-folding scores are "
          "committed as config/mpnn_log_odds.json together with the model commit and the "
          "weight checksum.  See README for how to recompute them.")


if __name__ == "__main__":
    main()
