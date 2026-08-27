"""Create a safe, machine-readable inventory of the PRONTO full archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iia_benchmark.data import audit_pronto_archive


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "data/public_datasets/pronto/PRONTO_benchmark_case_study.zip"
DEFAULT_OUTPUT = ROOT / "data/public_datasets/pronto/archive_inventory.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-crc", action="store_true")
    args = parser.parse_args()
    report = audit_pronto_archive(args.archive, verify_crc=args.verify_crc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["safe_to_extract"] and report["crc_failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
