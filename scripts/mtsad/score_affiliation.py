"""Re-score saved predictions under the affiliation metric family.

The CATCH paper reports the affiliation family (Aff-F, A-R) on the same five
datasets that the anchor papers report with point-adjusted F1, so a re-run can
only be checked against it after its predictions are converted into the same
family.  The conversion uses the reference implementation bundled with the
KDD'23 DCdetector release, fetched on demand into a scratch directory and never
vendored into this repository.

Input is the npz written by ``run_reproduction.py --save-scores``.  The threshold
is the harness's own reference rule: the (100 - anomaly_ratio) percentile of the
pooled train and test energies.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.evaluation.mtsad_metrics import adjusted_metrics  # noqa: E402

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
        opener = urllib.request.build_opener()
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )
        request = urllib.request.Request(
            base + source, headers={"User-Agent": "iia-benchmark/0.1"}
        )
        with opener.open(request, timeout=60) as response:
            destination.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--anomaly-ratio", type=float, required=True)
    parser.add_argument("--layout", default="window_flatten")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "tmp" / "mtsad_recon" / "refmetrics",
    )
    parser.add_argument("--repo", default="DAMO-DI-ML/KDD2023-DCdetector")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fetch_reference(
        args.reference_dir, repo=args.repo, revision=args.revision, proxy=args.proxy
    )
    sys.path.insert(0, str(args.reference_dir))
    generics = importlib.import_module("metrics.affiliation.generics")
    affiliation = importlib.import_module("metrics.affiliation.metrics")

    records = []
    for path in args.scores:
        payload = np.load(path)
        scores = payload[f"{args.layout}__test"]
        truth = payload[f"{args.layout}__truth"]
        train = payload[f"{args.layout}__train"]
        finite = np.isfinite(scores)
        scored_scores = scores[finite]
        scored_truth = truth[finite]
        pooled = np.concatenate([train[np.isfinite(train)], scored_scores])
        threshold = float(np.percentile(pooled, 100.0 - args.anomaly_ratio))
        prediction = (scored_scores > threshold).astype(int)
        events_pred = generics.convert_vector_to_events(prediction)
        events_gt = generics.convert_vector_to_events(scored_truth.astype(int))
        values = affiliation.pr_from_events(
            events_pred, events_gt, (0, len(prediction))
        )
        adjusted = adjusted_metrics(scored_truth, prediction.astype(bool))
        records.append(
            {
                "scores_file": path.name,
                "threshold": threshold,
                "scored_points": int(finite.sum()),
                "affiliation_precision": values["precision"],
                "affiliation_recall": values["recall"],
                "point_adjusted_f1": adjusted["f1"],
                "point_adjusted_precision": adjusted["precision"],
                "point_adjusted_recall": adjusted["recall"],
            }
        )
    print(json.dumps(records, indent=2))
    if args.out:
        args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
