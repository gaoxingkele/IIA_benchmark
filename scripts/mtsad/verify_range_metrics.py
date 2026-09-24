"""Compare the local range-based metric with the reference affiliation variant.

The two names cover different definitions from the same family:

* the local ``range_based_metrics(alpha=0, bias="flat")`` implements Tatbul et
  al.'s *range-based* overlap: precision of a predicted range is its best
  overlap fraction with a real range, recall of a real range is the fraction it
  is covered by predicted ranges;
* the *affiliation* variant bundled with the DCdetector release scores each
  point inside a predicted range by its positional affinity to the affiliated
  real range, so a partially overlapping prediction keeps partial credit.

The script downloads the reference implementation (never vendored into this
repository) and reports the disagreement on random binary series, so the choice
of variant is an explicit, measured decision rather than an assumption.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.evaluation.mtsad_metrics import range_based_metrics  # noqa: E402

REFERENCE_FILES = {
    "metrics/__init__.py": "",
    "metrics/affiliation/__init__.py": "",
    "metrics/affiliation/generics.py": "metrics/affiliation/generics.py",
    "metrics/affiliation/metrics.py": "metrics/affiliation/metrics.py",
    "metrics/affiliation/_affiliation_zone.py": "metrics/affiliation/_affiliation_zone.py",
    "metrics/affiliation/_integral_interval.py": "metrics/affiliation/_integral_interval.py",
    "metrics/affiliation/_single_ground_truth_event.py": "metrics/affiliation/_single_ground_truth_event.py",
}


def fetch_reference(target: Path, *, repo: str, revision: str, proxy: str | None) -> None:
    base = f"https://raw.githubusercontent.com/{repo}/{revision}/"
    for relative, source in REFERENCE_FILES.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size:
            continue
        if not source:
            destination.write_text("", encoding="utf-8")
            continue
        request = urllib.request.Request(
            base + source, headers={"User-Agent": "iia-benchmark/0.1"}
        )
        opener = urllib.request.build_opener()
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )
        with opener.open(request, timeout=60) as response:
            destination.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir", type=Path, default=ROOT / "tmp" / "mtsad_recon" / "refmetrics"
    )
    parser.add_argument("--repo", default="DAMO-DI-ML/KDD2023-DCdetector")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fetch_reference(
        args.reference_dir, repo=args.repo, revision=args.revision, proxy=args.proxy
    )
    sys.path.insert(0, str(args.reference_dir))
    generics = importlib.import_module("metrics.affiliation.generics")
    reference_metrics = importlib.import_module("metrics.affiliation.metrics")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.trials):
        length = int(rng.integers(200, 3000))
        truth = (rng.random(length) < 0.05).astype(int)
        prediction = (rng.random(length) < 0.04).astype(int)
        if truth.sum() == 0 or prediction.sum() == 0:
            continue
        reference = reference_metrics.pr_from_events(
            generics.convert_vector_to_events(prediction),
            generics.convert_vector_to_events(truth),
            (0, length),
        )
        ours = range_based_metrics(truth.astype(bool), prediction.astype(bool))
        rows.append(
            {
                "length": length,
                "local_precision": ours["precision"],
                "reference_precision": reference["precision"],
                "local_recall": ours["recall"],
                "reference_recall": reference["recall"],
            }
        )
    local_p = float(np.mean([row["local_precision"] for row in rows]))
    ref_p = float(np.mean([row["reference_precision"] for row in rows]))
    local_r = float(np.mean([row["local_recall"] for row in rows]))
    ref_r = float(np.mean([row["reference_recall"] for row in rows]))
    summary = {
        "trials": len(rows),
        "mean_local_precision": local_p,
        "mean_reference_precision": ref_p,
        "mean_local_recall": local_r,
        "mean_reference_recall": ref_r,
        "max_precision_gap": max(
            abs(row["local_precision"] - row["reference_precision"]) for row in rows
        ),
        "max_recall_gap": max(
            abs(row["local_recall"] - row["reference_recall"]) for row in rows
        ),
        "reference": f"{args.repo}@{args.revision}",
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
