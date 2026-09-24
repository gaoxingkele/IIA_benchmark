"""Re-score saved predictions at a POT (peaks-over-threshold) decision boundary.

TranAD's release thresholds its anomaly scores with POT instead of a percentile:
`src/pot.py` builds a `SPOT` object, fits it on the training scores, and takes
`mean(thresholds) * lm_d[1]`.  This script implements the documented static POT
rule - the one the SPOT library computes in its non-dynamic mode:

    t      = quantile(train, 1 - level)          # level = lm_d[0]
    xi, s  = GPD fit to (train[train > t] - t)
    th     = t + (s / xi) * ((risk * n / Nt) ** (-xi) - 1)
    score' = th * lm_d[1]

It is an implementation of the algorithm, not a binding to the `SPOT` package,
and that difference is reported rather than hidden.  The training scores come
from the same persisted file as the test scores, so no retraining is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.evaluation.mtsad_metrics import adjusted_metrics  # noqa: E402

# src/params.json of the TranAD release, per dataset (the single-value entries it
# uses for the datasets this artifact re-runs).
LM_PARAMS = {
    "smap": (0.98, 1.0),
    "msl": (0.999, 1.04),
    "swat": (0.993, 1.0),
    "smd": (0.99995, 1.06),
}


def pot_threshold(
    train_scores: np.ndarray, risk: float, level: float, multiplier: float
) -> float:
    from scipy.stats import genpareto

    finite = train_scores[np.isfinite(train_scores)]
    if finite.size < 100:
        raise ValueError("not enough training scores to fit POT")
    initial = float(np.quantile(finite, 1.0 - level))
    exceedances = finite[finite > initial] - initial
    n = finite.size
    exceedance_count = exceedances.size
    if exceedance_count < 10:
        return initial * multiplier
    xi, _, sigma = genpareto.fit(exceedances, floc=0.0)
    if abs(xi) < 1e-6:
        threshold = initial + sigma * np.log(risk * n / exceedance_count)
    else:
        threshold = initial + (sigma / xi) * (
            (risk * n / exceedance_count) ** (-xi) - 1
        )
    return float(threshold * multiplier)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(LM_PARAMS))
    parser.add_argument("--risk", type=float, default=1e-5)
    parser.add_argument("--layout", default="last_point")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    level, multiplier = LM_PARAMS[args.dataset]
    records = []
    for path in args.scores:
        payload = np.load(path)
        scores = payload[f"{args.layout}__test"]
        truth = payload[f"{args.layout}__truth"]
        train = payload[f"{args.layout}__train"]
        finite = np.isfinite(scores)
        scored_scores = scores[finite]
        scored_truth = truth[finite]
        threshold = pot_threshold(train, args.risk, level, multiplier)
        prediction = scored_scores > threshold
        metrics = adjusted_metrics(scored_truth, prediction)
        records.append(
            {
                "scores_file": path.name,
                "layout": args.layout,
                "pot_level": level,
                "pot_multiplier": multiplier,
                "pot_risk": args.risk,
                "threshold": threshold,
                "point_adjusted_f1": metrics["f1"],
                "point_adjusted_precision": metrics["precision"],
                "point_adjusted_recall": metrics["recall"],
                "scored_points": int(finite.sum()),
            }
        )
    print(json.dumps(records, indent=2))
    if args.out:
        args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
