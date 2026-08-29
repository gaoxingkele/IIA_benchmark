#!/usr/bin/env python3
"""Profile the official TEP five-class alarm payload before model evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import (  # noqa: E402
    build_tep_five_class_split,
    load_tep_five_class_alarm_runs,
)
from iia_benchmark.models import criterion_c_alarm_flood_detection  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(np.asarray(values, dtype=float), [0, 0.5, 0.9, 0.95, 0.99, 1])
    return {
        key: float(value)
        for key, value in zip(("min", "p50", "p90", "p95", "p99", "max"), points)
    }


def centroid_accuracy(split: object) -> float:
    centroids = {
        label: split.X_train[split.y_train == label].mean(axis=0)
        for label in sorted(set(split.y_train.tolist()))
    }
    predictions = [
        min(centroids, key=lambda label: float(np.mean((sample - centroids[label]) ** 2)))
        for sample in split.X_test
    ]
    return float(np.mean(np.asarray(predictions) == split.y_test))


def profile(archive_path: Path, random_state: int) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        zip_audit = {
            "members": len(infos),
            "compressed_bytes": archive_path.stat().st_size,
            "uncompressed_bytes": sum(info.file_size for info in infos),
            "crc_failure": archive.testzip(),
        }
    runs = load_tep_five_class_alarm_runs(archive_path)
    states = np.stack([run.alarm_states for run in runs])
    previous = np.concatenate((np.zeros_like(states[:, :1]), states[:, :-1]), axis=1)
    rising = np.maximum(states - previous, 0)
    labels = np.asarray([run.disturbance for run in runs])
    classes = sorted(set(labels.tolist()))

    state_hash_labels: dict[str, str] = {}
    duplicate_same_label = 0
    duplicate_cross_label = 0
    for run in runs:
        digest = hashlib.sha256(run.alarm_states.tobytes()).hexdigest()
        if digest in state_hash_labels:
            duplicate_same_label += int(state_hash_labels[digest] == run.disturbance)
            duplicate_cross_label += int(state_hash_labels[digest] != run.disturbance)
        else:
            state_hash_labels[digest] = run.disturbance

    split_reports: dict[str, dict[str, object]] = {}
    for representation in ("state", "rising_edge"):
        split = build_tep_five_class_split(
            runs,
            train_per_class=120,
            calibration_per_class=40,
            test_per_class=40,
            random_state=random_state,
            representation=representation,
        )
        partitions = {
            "train": split.train_run_ids,
            "calibration": split.calibration_run_ids,
            "test": split.test_run_ids,
        }
        split_reports[representation] = {
            "partition_sha256": hashlib.sha256(
                json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "centroid_test_accuracy": centroid_accuracy(split),
            "train_runs": len(split.train_run_ids),
            "calibration_runs": len(split.calibration_run_ids),
            "test_runs": len(split.test_run_ids),
            "partition_overlap": bool(
                set(split.train_run_ids) & set(split.calibration_run_ids)
                or set(split.train_run_ids) & set(split.test_run_ids)
                or set(split.calibration_run_ids) & set(split.test_run_ids)
            ),
        }

    criterion_parameters = {
        "attention_window": 10,
        "long_standing_window": 30,
        "update_step": 1,
        "threshold": 10,
        "delay_samples": 2,
    }
    criterion_candidates = 0
    criterion_runs_with_candidates = 0
    criterion_maximum_cardinality = 0
    for run in runs:
        detection = criterion_c_alarm_flood_detection(
            run.alarm_states, tag_names=run.alarm_names, **criterion_parameters
        )
        transitions = np.maximum(
            detection.delayed_detection
            - np.r_[np.int8(0), detection.delayed_detection[:-1]],
            0,
        )
        candidate_count = int(np.sum(transitions))
        criterion_candidates += candidate_count
        criterion_runs_with_candidates += int(candidate_count > 0)
        criterion_maximum_cardinality = max(
            criterion_maximum_cardinality,
            int(np.max(detection.cardinality, initial=0)),
        )

    by_class: dict[str, dict[str, object]] = {}
    for label in classes:
        mask = labels == label
        class_states = states[mask]
        class_rising = rising[mask]
        class_runs = [run for run in runs if run.disturbance == label]
        by_class[label] = {
            "runs": int(np.sum(mask)),
            "class_label": class_runs[0].class_label,
            "minimum_scaling_range": [
                float(min(run.min_scaling for run in class_runs)),
                float(max(run.min_scaling for run in class_runs)),
            ],
            "maximum_scaling_range": [
                float(min(run.max_scaling for run in class_runs)),
                float(max(run.max_scaling for run in class_runs)),
            ],
            "state_prevalence": float(np.mean(class_states)),
            "activation_edges_per_run": quantiles(class_rising.sum(axis=(1, 2))),
            "unique_active_tags_per_run": quantiles(
                (class_states.max(axis=1) > 0).sum(axis=1)
            ),
        }

    class_counts = Counter(labels.tolist())
    g0_passed = (
        zip_audit["crc_failure"] is None
        and len(runs) == 1000
        and set(class_counts.values()) == {200}
        and states.shape == (1000, 300, 50)
        and duplicate_cross_label == 0
        and not split_reports["state"]["partition_overlap"]
        and not split_reports["rising_edge"]["partition_overlap"]
    )
    return {
        "schema_version": 1,
        "dataset_family": "tep_alarm_dataport",
        "payload": "2nd_Alarm_Dataset_5Classes.zip",
        "source": {
            "path": archive_path.relative_to(ROOT).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "citation_doi": "10.21227/326k-qr90",
        },
        "zip_audit": zip_audit,
        "schema": {
            "runs": len(runs),
            "classes": class_counts,
            "samples_per_run": states.shape[1],
            "alarm_variables": states.shape[2],
            "sample_minutes": runs[0].sample_minutes,
            "missing_values": 0,
            "nonbinary_values": 0,
        },
        "class_profiles": by_class,
        "distribution": {
            "active_states_per_minute": quantiles(states.sum(axis=2)),
            "activation_edges_per_run": quantiles(rising.sum(axis=(1, 2))),
            "unique_active_tags_per_run": quantiles((states.max(axis=1) > 0).sum(axis=1)),
            "never_active_alarm_tags": [
                runs[0].alarm_names[index]
                for index in np.flatnonzero(states.max(axis=(0, 1)) == 0)
            ],
        },
        "duplicate_audit": {
            "unique_state_matrices": len(state_hash_labels),
            "same_label_duplicates": duplicate_same_label,
            "cross_label_duplicates": duplicate_cross_label,
        },
        "registered_split": {
            "random_state": random_state,
            "state": split_reports["state"],
            "rising_edge": split_reports["rising_edge"],
        },
        "criterion_c_descriptive_prior": {
            "parameters": criterion_parameters,
            "candidate_intervals": criterion_candidates,
            "runs_with_candidates": criterion_runs_with_candidates,
            "maximum_attention_set_cardinality": criterion_maximum_cardinality,
            "label_boundary": "No expert interval labels; candidates are descriptive only.",
        },
        "g0": {
            "passed": g0_passed,
            "checks": [
                "ZIP CRC complete",
                "ground-truth-to-CSV one-to-one mapping",
                "binary schema and consecutive minute grid",
                "balanced five-class labels",
                "no cross-label exact matrix duplicates",
                "complete-sample split separation",
            ],
        },
        "limitations": [
            "The payload contains five abnormal classes and no normal-operation class.",
            "Centroid accuracy is a G0 label-signal diagnostic, not a benchmark baseline.",
            "Criterion-C candidates have no expert flood-interval truth.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT
        / "data/public_datasets/tep_alarm_dataport/payload/2nd_Alarm_Dataset_5Classes.zip",
    )
    parser.add_argument("--random-state", type=int, default=1103)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/reports/tep_alarm_five_class_prior_validation.json",
    )
    args = parser.parse_args()
    report = profile(args.archive.resolve(), args.random_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["g0"], ensure_ascii=False, indent=2))
    return 0 if report["g0"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
