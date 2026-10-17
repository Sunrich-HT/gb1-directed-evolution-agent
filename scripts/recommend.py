#!/usr/bin/env python
"""Thin wrapper so the recommender can be run from a checkout without installing.

Installed users get the same entry point as ``gb1-recommend`` (see pyproject.toml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.cli import main

if __name__ == "__main__":
    main()
