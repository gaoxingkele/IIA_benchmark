"""Turn reproduction run records into markdown evidence tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["SMD", "MSL", "SMAP", "SWaT", "PSM"]
MODEL_LABELS = {
    "pca": "PCA",
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
    "isolation_forest": "IsolationForest",
    "ocsvm": "OCSVM",
    "usad": "USAD",
    "anomaly_transformer": "AnomalyTransformer",
}


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            records.extend(payload["records"])
        elif isinstance(payload, dict):
            records.append(payload)
        else:
            records.extend(payload)
    return records


def aggregate(records: list[dict], metric: str) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        protocol = record["protocols"].get("tslib_reference")
        if not protocol or metric not in protocol:
            continue
        grouped[(record["model"], record["dataset"])].append(protocol[metric])
    return grouped


def fmt(values: list[float]) -> str:
    if not values:
        return "-"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {float(np.std(values)):.4f}"


def table_for(records: list[dict], metric: str, caption: str) -> str:
    grouped = aggregate(records, metric)
    models = [model for model in MODEL_LABELS if any(key[0] == model for key in grouped)]
    lines = [f"**{caption}**", "", "| Method | " + " | ".join(DATASETS) + " |", "|---" * (len(DATASETS) + 1) + "|"]
    for model in models:
        cells = [fmt(grouped.get((model, dataset.lower()), [])) for dataset in DATASETS]
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def protocol_table(records: list[dict]) -> str:
    protocols = sorted({name for record in records for name in record["protocols"]})
    lines = [
        "**Same detector, four protocols (headline F1 per protocol)**",
        "",
        "| Dataset | Method | " + " | ".join(protocols) + " |",
        "|---" * (len(protocols) + 2) + "|",
    ]
    seen: set[tuple[str, str]] = set()
    for record in sorted(records, key=lambda item: (item["dataset"], item["model"], item["seed"])):
        key = (record["dataset"], record["model"])
        if key in seen:
            continue
        seen.add(key)
        cells = []
        for name in protocols:
            metrics = record["protocols"].get(name)
            cells.append(f"{metrics['reported_protocol_f1']:.4f}" if metrics else "-")
        model = MODEL_LABELS.get(record["model"], record["model"])
        lines.append(f"| {record['dataset'].upper()} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(records: list[dict]) -> str:
    reported_path = ROOT / "knowledge_base" / "literature" / "mtsad_reported_results.json"
    reported = json.loads(reported_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], float] = {}
    for source in reported["sources"]:
        for method, rows in source["rows"].items():
            for dataset, metrics in rows.items():
                value = metrics.get("f1")
                if value is None:
                    continue
                scale = 100.0 if value > 1.5 else 1.0
                lookup[(method, dataset)] = value / scale
    grouped = aggregate(records, "adjusted_f1")
    lines = [
        "**Re-run (reference protocol) versus published numbers**",
        "",
        "| Method | Dataset | Re-run adjusted F1 | Published F1 | Source |",
        "|---|---|---|---|---|",
    ]
    for (method, dataset), value in sorted(lookup.items()):
        ours = grouped.get((method, dataset.lower()))
        lines.append(
            f"| {MODEL_LABELS.get(method, method)} | {dataset} | {fmt(ours) if ours else 'not re-run'} | "
            f"{value:.4f} | {reported['sources'][0]['paper_id'] if method == 'anomaly_transformer' else 'registered paper table'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Reproduction results")
    args = parser.parse_args()
    records = load_records(args.records)
    sections = [
        f"# {args.title}",
        "",
        f"Generated {date.today().isoformat()} from {len(records)} run records.",
        "Every number below is produced by `scripts/mtsad/run_reproduction.py`; the",
        "protocol named in each row is the one declared in",
        "`configs/experiments/mtsad_reproduction.json`.",
        "",
        table_for(records, "adjusted_f1", "Point-adjusted F1 (reference protocol: percentile threshold + point adjustment)"),
        "",
        table_for(records, "pointwise_f1", "Point-wise F1 at the same threshold (no point adjustment)"),
        "",
        table_for(records, "range_f1", "Range-based F1 (alpha = 0, flat bias)"),
        "",
        protocol_table(records),
        "",
        comparison_table(records),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.out} from {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
