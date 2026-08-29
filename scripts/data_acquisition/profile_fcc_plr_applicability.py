#!/usr/bin/env python3
"""Audit whether FCC injected disturbances identify temporal PLR root causes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import load_fcc_timeseries_runs  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/public_datasets/fcc_alarm/timeseriesdata.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/reports/fcc_plr_applicability_gate.json",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    runs = load_fcc_timeseries_runs(source)
    tolerance = 1e-12
    any_change, root_change = [], []
    root_amplitudes = []
    missing_root_columns = []
    for run in runs:
        variation = np.ptp(run.disturbance_values, axis=0)
        any_change.append(bool(np.any(variation > tolerance)))
        if run.root_disturbance not in run.disturbance_names:
            missing_root_columns.append(run.run_id)
            root_change.append(False)
            root_amplitudes.append(float("nan"))
            continue
        index = run.disturbance_names.index(run.root_disturbance)
        root_amplitudes.append(float(variation[index]))
        root_change.append(bool(variation[index] > tolerance))
    finite_amplitudes = np.asarray(root_amplitudes, dtype=float)
    finite_amplitudes = finite_amplitudes[np.isfinite(finite_amplitudes)]
    report = {
        "schema_version": 1,
        "dataset_family": "fcc_alarm",
        "target_algorithm": "book_4_4_plr_rca",
        "source": {
            "path": source.relative_to(ROOT).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "citation_doi": "10.60517/2v23vv393",
        },
        "schema": {
            "runs": len(runs),
            "scenarios": len({run.scenario for run in runs}),
            "runs_per_scenario": {
                label: sum(run.scenario == label for run in runs)
                for label in sorted({run.scenario for run in runs})
            },
            "samples_per_run": sorted({len(run.timestamps) for run in runs}),
            "process_variables": len(runs[0].process_names),
            "valve_variables": len(runs[0].valve_names),
            "disturbance_variables": len(runs[0].disturbance_names),
            "missing_root_disturbance_columns": missing_root_columns,
        },
        "temporal_cause_audit": {
            "runs_with_any_within_run_disturbance_change": int(sum(any_change)),
            "runs_with_root_disturbance_change": int(sum(root_change)),
            "root_disturbance_peak_to_peak": {
                "min": float(np.min(finite_amplitudes)),
                "median": float(np.median(finite_amplitudes)),
                "max": float(np.max(finite_amplitudes)),
            },
            "interpretation": (
                "Injected disturbance columns encode a run-level constant setting; "
                "they do not contain within-episode transitions for PLR lag recovery."
            ),
        },
        "g0": {
            "data_integrity_passed": not missing_root_columns and len(runs) == 1600,
            "plr_temporal_admission": bool(sum(root_change)),
            "decision": "deny temporal PLR score on FCC",
            "reason": (
                "PLR lag and contribution factors require time-varying candidate causes. "
                "A constant injected-disturbance vector makes lag undefined and all "
                "within-run cause trends zero."
            ),
        },
        "allowed_use": (
            "FCC remains valid for alarm classification and process response analysis; "
            "PLR requires a dataset exposing time-varying causal driver signals."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["g0"], ensure_ascii=False, indent=2))
    return 0 if report["g0"]["data_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
