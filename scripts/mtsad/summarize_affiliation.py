"""Join the affiliation-family re-scores with CATCH's published Aff-F column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "ara_mtsad" / "evidence" / "tables")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    published = json.loads(
        (ROOT / "knowledge_base" / "literature" / "mtsad_reported_affiliation.json").read_text(
            encoding="utf-8"
        )
    )
    columns = published["sources"][0]["columns"]
    published_rows = published["sources"][0]["rows"]
    index = {name: position for position, name in enumerate(columns)}

    runs = {
        "timesnet": {
            "MSL": "affiliation_timesnet_msl.json",
            "PSM": "affiliation_timesnet_psm.json",
            "SMD": "affiliation_timesnet_smd.json",
            "SMAP": "affiliation_timesnet_smap.json",
            "SWaT": "affiliation_timesnet_swat.json",
        },
        "dcdetector": {
            "MSL": "affiliation_dcdetector_msl.json",
            "PSM": "affiliation_dcdetector_psm.json",
            "SMD": "affiliation_dcdetector_smd.json",
            "SMAP": "affiliation_dcdetector_smap.json",
            "SWaT": "affiliation_dcdetector_swat.json",
        },
        "anomaly_transformer": {},
    }
    # The Anomaly Transformer conversion was produced by the earlier turn and is
    # recorded in the prose table of affiliation_family_check.md; it is added here
    # from its own JSON when present.
    anomaly_json = args.evidence_dir / "affiliation_anomaly_transformer_msl.json"
    if anomaly_json.is_file():
        runs["anomaly_transformer"]["MSL"] = anomaly_json.name

    lines = [
        "**Affiliation-family comparison (generated)**",
        "",
        "| Model | Dataset | affiliation P | affiliation R | affiliation F1 | CATCH Aff-F | delta |",
        "|---|---|---|---|---|---|---|",
    ]
    tally: list[float] = []
    for model, datasets in runs.items():
        for dataset, filename in sorted(datasets.items()):
            path = args.evidence_dir / filename
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))[0]
            f1 = harmonic(payload["affiliation_precision"], payload["affiliation_recall"])
            target = published_rows[dataset]["affiliation_f1"][index[model]]
            delta = abs(f1 - target)
            tally.append(delta)
            lines.append(
                f"| {model} | {dataset} | {payload['affiliation_precision']:.4f} | "
                f"{payload['affiliation_recall']:.4f} | {f1:.4f} | {target:.3f} | {delta:.4f} |"
            )
    lines += [
        "",
        f"Comparisons: {len(tally)}. Mean absolute delta {sum(tally)/len(tally):.4f}. "
        f"Worst {max(tally):.4f}. Best {min(tally):.4f}.",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
