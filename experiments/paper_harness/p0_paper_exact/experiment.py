#!/usr/bin/env python3
"""Entry point for P0 author-capsule Paper-Exact runs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts/paper_exact.py"), *sys.argv[1:]], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
