"""Threshold-free AUROC and AUPRC for a saved score file.

Some registered papers report ranking metrics rather than a thresholded F1 -
GCAD's Table 2 is AUROC/AUPRC - so their numbers can only be checked after the
re-run's scores are scored the same way. Because these metrics are threshold-free
they also remove the threshold rule as a confound: the comparison is about the
ranking the model produces, not about where a boundary is placed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--layout", default="last_point")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score

    records = []
    for path in args.scores:
        payload = np.load(path)
        scores = payload[f"{args.layout}__test"]
        truth = payload[f"{args.layout}__truth"].astype(int)
        finite = np.isfinite(scores)
        scores = scores[finite]
        truth = truth[finite]
        records.append(
            {
                "scores_file": path.name,
                "layout": args.layout,
                "auroc": float(roc_auc_score(truth, scores)),
                "auprc": float(average_precision_score(truth, scores)),
                "scored_points": int(finite.sum()),
                "anomaly_rate": float(truth.mean()),
            }
        )
    print(json.dumps(records, indent=2))
    if args.out:
        args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
