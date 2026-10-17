#!/usr/bin/env python
"""Fail the build if the tree carries a credential, an absolute path or build junk.

Run locally with `python scripts/check_repo_hygiene.py`; CI runs the same script, so
what passes here is what passes there.  Vendored third-party code is checked for
credentials but not for paths: its defaults belong to upstream and are not ours to
rewrite.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CREDENTIALS = {
    "OpenAI-style key": r"sk-[A-Za-z0-9]{16,}",
    "NVIDIA key": r"nvapi-[A-Za-z0-9_\-]{18,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "AWS key id": r"AKIA[0-9A-Z]{16}",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
PATHS = {
    "home directory": r"/(?:Users|home)/[A-Za-z0-9._-]+",
    "cluster scratch": r"/cadd_data[A-Za-z0-9/._-]*",
    "sandbox workspace": r"\.[A-Za-z0-9-]+/(?:orgs|workspaces)/[0-9a-f][0-9a-f-]{7,}[A-Za-z0-9/._-]*",
    "hostname or address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}
JUNK = r"(__pycache__|\.pyc$|\.DS_Store$|\.aux$|\.synctex\.gz$|\.ipynb_checkpoints)"
VENDOR_PREFIX = "vendor/"
MAX_BYTES = 8_000_000


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return out.stdout.split()


def main() -> int:
    files = tracked_files()
    problems: list[str] = []

    for name in files:
        if re.search(JUNK, name):
            problems.append(f"build artefact committed: {name}")

    for name in files:
        path = Path(name)
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for label, pattern in CREDENTIALS.items():
            if re.search(pattern, text):
                problems.append(f"{label} in {name}")
        if name.startswith(VENDOR_PREFIX):
            continue
        for label, pattern in PATHS.items():
            for hit in set(re.findall(pattern, text)):
                problems.append(f"{label} in {name}: {hit}")

    print(f"checked {len(files)} tracked files")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in sorted(set(problems)):
            print(f"  - {p}")
        return 1
    print("no credentials, no absolute paths, no build artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
