#!/usr/bin/env python
"""Small-scale, zero-network reproduction of the whole pipeline.

Runs every arm with the deterministic mock agent backend on a reduced budget so
that a reviewer can verify the machinery end to end in about a minute without
an API key.  The numbers it produces are an engineering check, not a scientific
result -- the mock backend does not reason.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "four_mutations_full_data.csv.zip")
    if not Path(data).exists():
        raise SystemExit(
            f"landscape file not found at {data}\n"
            "download it once with:\n"
            "  curl -L -o data/four_mutations_full_data.csv.zip \\\n"
            "    https://raw.githubusercontent.com/J-SNACKKB/FLIP/"
            "62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/four_mutations_full_data.csv.zip"
        )
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
        "--data", data, "--out", str(ROOT / "results" / "reproduce_small"),
        "--seeds", "301", "--rounds", "2", "--batch-size", "48", "--backend", "mock",
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("\nOK: full chain ran offline (split -> model selection -> agent loop -> oracle -> metrics)")


if __name__ == "__main__":
    main()
