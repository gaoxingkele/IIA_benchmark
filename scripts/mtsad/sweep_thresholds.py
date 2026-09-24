"""Sweep decision thresholds on a saved score file and report the best F1-PA.

This separates two explanations for a reproduction gap:

* if an oracle threshold reaches the published number, the gap is threshold
  placement (the percentile rule landing badly on this score distribution);
* if no threshold does, the score ranking itself is weaker than the published
  one and the gap is a model/score difference.

Use `run_reproduction.py --save-scores` to produce the input file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.evaluation.mtsad_metrics import (  # noqa: E402
    adjusted_metrics,
    pointwise_metrics,
)


def sweep(scores: np.ndarray, truth: np.ndarray, points: int) -> dict:
    mask = np.isfinite(scores)
    scores, truth = scores[mask], truth[mask]
    thresholds = np.quantile(scores, np.linspace(0.5, 0.9999, points))
    best = {"adjusted_f1": -1.0}
    for threshold in thresholds:
        prediction = scores > threshold
        adjusted = adjusted_metrics(truth, prediction)
        if adjusted["f1"] > best["adjusted_f1"]:
            point = pointwise_metrics(truth, prediction)
            best = {
                "adjusted_f1": adjusted["f1"],
                "adjusted_precision": adjusted["precision"],
                "adjusted_recall": adjusted["recall"],
                "pointwise_f1": point["f1"],
                "threshold": float(threshold),
                "kept_fraction": float((scores <= threshold).mean()),
            }
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--layout", default="window_flatten")
    args = parser.parse_args()
    payload = np.load(args.scores)
    scores = payload[f"{args.layout}__test"]
    truth = payload[f"{args.layout}__truth"]
    reference = payload[f"{args.layout}__train"]
    result = {
        "layout": args.layout,
        "oracle": sweep(scores, truth, args.points),
        "reference_percentile_threshold": float(
            np.percentile(
                np.concatenate([reference[np.isfinite(reference)], scores[np.isfinite(scores)]]),
                99.0,
            )
        ),
        "scored_points": int(np.isfinite(scores).sum()),
    }
    threshold = result["reference_percentile_threshold"]
    prediction = np.nan_to_num(scores, nan=-np.inf) > threshold
    metrics = adjusted_metrics(truth[np.isfinite(scores)], prediction[np.isfinite(scores)])
    result["at_reference_threshold"] = {
        "adjusted_f1": metrics["f1"],
        "adjusted_precision": metrics["precision"],
        "adjusted_recall": metrics["recall"],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
