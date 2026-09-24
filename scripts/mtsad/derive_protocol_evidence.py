"""Derive the protocol-sensitivity evidence table directly from run records."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "pca": "PCA",
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
    "isolation_forest": "IsolationForest",
    "ocsvm": "OCSVM",
    "usad": "USAD (non-adversarial)",
    "anomaly_transformer": "AnomalyTransformer",
}


def load(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if path.is_file():
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    return records


def mean_metric(records: list[dict], dataset: str, model: str, protocol: str, field: str):
    values = [
        record["protocols"][protocol][field]
        for record in records
        if record["dataset"] == dataset
        and record["model"] == model
        and protocol in record["protocols"]
    ]
    return float(np.mean(values)) if values else None


def main() -> int:
    run_root = ROOT / "experiments" / "runs" / "mtsad_reproduction"
    # Every run directory, so the table tracks the whole matrix rather than the
    # first two batches it was written for.
    paths = sorted(
        path for path in run_root.glob("*/records.json") if path.parent.name != "smoke"
    )
    records = load(paths)
    if not records:
        raise SystemExit("no run records found")
    pairs = sorted({(record["dataset"], record["model"]) for record in records})

    lines = [
        "# Protocol sensitivity",
        "",
        f"Derived on {date.today().isoformat()} by `scripts/mtsad/derive_protocol_evidence.py`",
        f"from {len(records)} run records in `experiments/runs/mtsad_reproduction/`.",
        "Each detector is scored once per run; the rows below differ only in the",
        "declared protocol, so the spread is attributable to the protocol.",
        "",
        "## Adjusted versus point-wise F1 (reference threshold rule)",
        "",
        "| Dataset | Method | adjusted F1 | point-wise F1 | range F1 | ratio adj/point |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, model in pairs:
        adjusted = mean_metric(records, dataset, model, "tslib_reference", "adjusted_f1")
        pointwise = mean_metric(records, dataset, model, "tslib_reference", "pointwise_f1")
        ranges = mean_metric(records, dataset, model, "tslib_reference", "range_f1")
        if adjusted is None or pointwise is None:
            continue
        ratio = adjusted / pointwise if pointwise else float("inf")
        lines.append(
            f"| {dataset.upper()} | {LABELS.get(model, model)} | {adjusted:.4f} | "
            f"{pointwise:.4f} | {ranges:.4f} | {ratio:.1f} |"
        )

    lines += [
        "",
        "## Threshold rule sensitivity (point adjustment on in both columns)",
        "",
        "| Dataset | Method | percentile threshold adjusted F1 | oracle threshold adjusted F1 |",
        "|---|---|---|---|",
    ]
    for dataset, model in pairs:
        percentile = mean_metric(records, dataset, model, "tslib_reference", "adjusted_f1")
        oracle = mean_metric(records, dataset, model, "best_f1_adjusted", "adjusted_f1")
        if percentile is None or oracle is None:
            continue
        lines.append(
            f"| {dataset.upper()} | {LABELS.get(model, model)} | {percentile:.4f} | {oracle:.4f} |"
        )

    lines += [
        "",
        "## Layout sensitivity for windowed detectors",
        "",
        "| Dataset | Method | window_flatten adjusted F1 | last_point adjusted F1 |",
        "|---|---|---|---|",
    ]
    for dataset, model in pairs:
        flat = mean_metric(records, dataset, model, "best_f1_adjusted", "adjusted_f1")
        last = mean_metric(records, dataset, model, "per_point_adjusted", "adjusted_f1")
        if flat is None or last is None:
            continue
        if abs(flat - last) < 1e-12:
            continue
        lines.append(
            f"| {dataset.upper()} | {LABELS.get(model, model)} | {flat:.4f} | {last:.4f} |"
        )

    lines += [
        "",
        "## Cross-dataset operating point (PCA, reference protocol recall)",
        "",
        "| Dataset | labelled anomaly rate | recall |",
        "|---|---|---|",
    ]
    rates = {}
    for record in records:
        rates.setdefault(record["dataset"], record["test_anomaly_rate"])
    for dataset in sorted(rates):
        recall = mean_metric(records, dataset, "pca", "tslib_reference", "pointwise_recall")
        if recall is None:
            continue
        lines.append(f"| {dataset.upper()} | {rates[dataset]:.4f} | {recall:.4f} |")

    lines.append("")
    out = ROOT / "ara_mtsad" / "evidence" / "tables" / "protocol_sensitivity.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} from {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
