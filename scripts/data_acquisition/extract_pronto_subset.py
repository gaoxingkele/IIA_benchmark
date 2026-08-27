"""Safely extract the bounded PRONTO preprocessed validation subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iia_benchmark.data import extract_pronto_members


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/public_datasets/pronto/PRONTO_benchmark_case_study.zip"
DESTINATION = ROOT / "data/public_datasets/pronto/extracted"
PREPROCESSED_PREFIX = (
    "PRONTO benchmark case study/Pre-processed data/"
    "Aligned and labelled alarm and process data"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--maximum-total-mib", type=int, default=64)
    args = parser.parse_args()
    paths = extract_pronto_members(
        args.archive,
        args.destination,
        prefixes=tuple(args.prefix or [PREPROCESSED_PREFIX]),
        maximum_total_bytes=args.maximum_total_mib * 1024 * 1024,
    )
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    print(json.dumps({"files": len(records), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
