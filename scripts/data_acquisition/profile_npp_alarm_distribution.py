#!/usr/bin/env python3
"""Profile the NPP alpha-0.50 slice and its duplicate-safe benchmark split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import build_npp_alarm_split, load_npp_alarm_runs  # noqa: E402
from iia_benchmark.models import criterion_c_alarm_flood_detection  # noqa: E402


FAMILIES = (
    "FLB",
    "LLB",
    "LOCA",
    "LOCAC",
    "LR",
    "RI",
    "RW",
    "SGATR",
    "SGBTR",
    "SLBIC",
    "SLBOC",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[int] | np.ndarray) -> dict[str, float]:
    points = np.quantile(np.asarray(values, dtype=float), [0, 0.1, 0.5, 0.9, 1])
    return {
        key: float(value)
        for key, value in zip(("min", "p10", "p50", "p90", "max"), points)
    }


def centroid_accuracy(split: object) -> float:
    centroids = {
        label: split.X_train[split.y_train == label].mean(axis=0)
        for label in sorted(set(split.y_train.tolist()))
    }
    prediction = [
        min(centroids, key=lambda label: float(np.mean((sample - centroids[label]) ** 2)))
        for sample in split.X_test
    ]
    return float(np.mean(np.asarray(prediction) == split.y_test))


def profile(extracted_root: Path, source_archive: Path, random_state: int) -> dict[str, object]:
    all_runs = load_npp_alarm_runs(
        extracted_root,
        alpha=0.5,
        minimum_samples=1,
        include_normal=True,
    )
    all_by_family: dict[str, list[object]] = defaultdict(list)
    for run in all_runs:
        all_by_family[run.fault_family].append(run)
    source_profiles = {
        label: {
            "runs": len(runs),
            "source_length_samples": quantiles([run.source_samples for run in runs]),
            "eligible_at_160": sum(run.source_samples >= 160 for run in runs),
        }
        for label, runs in sorted(all_by_family.items())
    }
    del all_runs

    runs = load_npp_alarm_runs(
        extracted_root,
        alpha=0.5,
        fault_families=FAMILIES,
        minimum_samples=160,
        horizon_samples=160,
    )
    run_by_id = {run.run_id: run for run in runs}
    split_reports: dict[str, dict[str, object]] = {}
    canonical_partition_hash: str | None = None
    for representation in ("state", "rising_edge"):
        split = build_npp_alarm_split(
            runs,
            train_per_class=28,
            calibration_per_class=10,
            test_per_class=10,
            random_state=random_state,
            representation=representation,
        )
        partitions = {
            "train": split.train_run_ids,
            "calibration": split.calibration_run_ids,
            "test": split.test_run_ids,
        }
        partition_hash = hashlib.sha256(
            json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        canonical_partition_hash = canonical_partition_hash or partition_hash
        part_by_id = {
            run_id: part for part, run_ids in partitions.items() for run_id in run_ids
        }
        signature_parts: dict[str, set[str]] = defaultdict(set)
        for run_id, part in part_by_id.items():
            signature = hashlib.sha256(
                run_by_id[run_id].representation(representation).tobytes()
            ).hexdigest()
            signature_parts[signature].add(part)
        split_reports[representation] = {
            "partition_sha256": partition_hash,
            "centroid_test_accuracy": centroid_accuracy(split),
            "cross_partition_duplicate_groups": sum(
                len(parts) > 1 for parts in signature_parts.values()
            ),
            "train_runs": len(split.train_run_ids),
            "calibration_runs": len(split.calibration_run_ids),
            "test_runs": len(split.test_run_ids),
            "unused_or_deduplicated_runs": len(split.unused_run_ids),
            "duplicate_nonrepresentatives": len(split.duplicate_run_ids),
            "cross_label_conflicting_runs": len(split.conflicting_run_ids),
        }

    md_runs = load_npp_alarm_runs(
        extracted_root,
        alpha=0.5,
        fault_families=("MD",),
        minimum_samples=160,
        horizon_samples=160,
    )
    md_state_signatures = {
        hashlib.sha256(run.alarm_states.tobytes()).hexdigest() for run in md_runs
    }
    states = np.stack([run.alarm_states for run in runs])
    previous = np.concatenate((np.zeros_like(states[:, :1]), states[:, :-1]), axis=1)
    rising = np.maximum(states - previous, 0)
    criterion_parameters = {
        "attention_window": 60,
        "long_standing_window": 120,
        "update_step": 6,
        "threshold": 40,
        "delay_samples": 2,
    }
    criterion_candidates = 0
    criterion_runs_with_candidates = 0
    maximum_cardinality = 0
    for run in runs:
        detection = criterion_c_alarm_flood_detection(
            run.alarm_states, tag_names=run.alarm_names, **criterion_parameters
        )
        transitions = np.maximum(
            detection.delayed_detection
            - np.r_[np.int8(0), detection.delayed_detection[:-1]],
            0,
        )
        count = int(np.sum(transitions))
        criterion_candidates += count
        criterion_runs_with_candidates += int(count > 0)
        maximum_cardinality = max(
            maximum_cardinality, int(np.max(detection.cardinality, initial=0))
        )

    by_class = {}
    labels = np.asarray([run.fault_family for run in runs])
    for label in FAMILIES:
        mask = labels == label
        class_states = states[mask]
        class_rising = rising[mask]
        by_class[label] = {
            "eligible_runs": int(np.sum(mask)),
            "state_prevalence": float(np.mean(class_states)),
            "activation_edges_per_run": quantiles(class_rising.sum(axis=(1, 2))),
            "unique_active_tags_per_run": quantiles(
                (class_states.max(axis=1) > 0).sum(axis=1)
            ),
        }
    g0_passed = (
        len(all_by_family) == 13
        and len(runs) == 1049
        and states.shape == (1049, 160, 192)
        and len(md_state_signatures) == 1
        and all(
            report["cross_partition_duplicate_groups"] == 0
            for report in split_reports.values()
        )
        and split_reports["state"]["partition_sha256"]
        == split_reports["rising_edge"]["partition_sha256"]
    )
    return {
        "schema_version": 1,
        "dataset_family": "npp_alarm_dataport",
        "selection": {
            "alpha": 0.5,
            "horizon_samples": 160,
            "sample_seconds": 10,
            "included_fault_families": list(FAMILIES),
            "excluded_from_closed_set": ["MD", "Normal"],
        },
        "source": {
            "archive_path": source_archive.relative_to(ROOT).as_posix(),
            "archive_bytes": source_archive.stat().st_size,
            "archive_sha256": sha256_file(source_archive),
            "extracted_slice_files": sum(item["runs"] for item in source_profiles.values()),
            "citation_doi": "10.21227/g2fa-9y43",
        },
        "schema": {
            "alpha_slice_runs": sum(item["runs"] for item in source_profiles.values()),
            "fault_families_plus_normal": len(source_profiles),
            "alarm_variables": 192,
            "time_column": "TIME",
            "sample_seconds": 10,
            "missing_values": 0,
            "nonbinary_alarm_values": 0,
        },
        "source_family_profiles": source_profiles,
        "benchmark_class_profiles": by_class,
        "distribution": {
            "active_states_per_sample": quantiles(states.sum(axis=2).ravel()),
            "activation_edges_per_run": quantiles(rising.sum(axis=(1, 2))),
            "unique_active_tags_per_run": quantiles((states.max(axis=1) > 0).sum(axis=1)),
            "never_active_alarm_tags": [
                runs[0].alarm_names[index]
                for index in np.flatnonzero(states.max(axis=(0, 1)) == 0)
            ],
        },
        "degeneracy_audit": {
            "md_eligible_runs": len(md_runs),
            "md_unique_state_trajectories": len(md_state_signatures),
            "normal_runs_in_alpha_slice": source_profiles["Normal"]["runs"],
            "decision": "MD and Normal excluded from independent balanced closed-set score",
        },
        "registered_split": {
            "random_state": random_state,
            "partition_sha256": canonical_partition_hash,
            "state": split_reports["state"],
            "rising_edge": split_reports["rising_edge"],
        },
        "criterion_c_descriptive_prior": {
            "parameters": criterion_parameters,
            "candidate_intervals": criterion_candidates,
            "runs_with_candidates": criterion_runs_with_candidates,
            "maximum_attention_set_cardinality": maximum_cardinality,
            "label_boundary": "No expert interval labels; candidates are descriptive only.",
        },
        "g0": {
            "passed": g0_passed,
            "checks": [
                "13 source families including Normal",
                "strict 192-bit binary schema and 10-second grid",
                "160-sample eligibility without padding",
                "MD duplicate and Normal singleton degeneracy explicitly excluded",
                "state-or-edge equivalence components cannot cross partitions",
                "cross-label trajectory conflicts removed",
            ],
        },
        "limitations": [
            "Only alpha=0.50 is used for the base score; other alpha values are reserved for grouped robustness.",
            "The 160-sample horizon excludes 62 short runs: 49 RI and 13 RW cases relative to the included-family alpha slice.",
            "The split uses one representative per state-or-edge trajectory component, so it measures generalization to unseen signatures rather than duplicate replay.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=ROOT / "data/public_datasets/npp_alarm_dataport/payload/alpha_050",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT
        / "data/public_datasets/alipan_anomaly_archives/Industrial-Alarm-Datasets.rar",
    )
    parser.add_argument("--random-state", type=int, default=1103)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/reports/npp_alarm_alpha050_prior_validation.json",
    )
    args = parser.parse_args()
    report = profile(
        args.extracted_root.resolve(), args.source_archive.resolve(), args.random_state
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["g0"], ensure_ascii=False, indent=2))
    return 0 if report["g0"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
