#!/usr/bin/env python3
"""Validate the independent ConE conformal layer on the author's frozen folds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
CONFIG_PATH = ROOT / "configs/experiments/p0_cone_independent_same_fold.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
EXPORT = ROOT / CONFIG["capsule_export"]
CODE = EXPORT / "code"
CACHE = ROOT / CONFIG["runtime_cache"]
OUTPUT = ROOT / CONFIG["output_root"]
REFERENCE_PATH = ROOT / CONFIG["author_reference"]
RUNTIME_CACHE = ROOT / "tmp/paper_exact_runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(RUNTIME_CACHE / "pycache")
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE / "numba"))


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    os.replace(temporary, path)


def prefixes() -> tuple[int, ...]:
    values = CONFIG["prefix_minutes"]
    return tuple(range(values["start"], values["stop"] + 1, values["step"]))


def sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def load_paper_grid_module() -> Any:
    path = PROJECT / "paper_grid.py"
    spec = importlib.util.spec_from_file_location("p0_paper_grid_for_independent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_balanced_metrics(
    prediction_sets: list[frozenset[Any]], y_true: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(y_true)
    cardinality = np.asarray([len(values) for values in prediction_sets])
    coverage: list[float] = []
    set_size: list[float] = []
    singleton: list[float] = []
    empty: list[float] = []
    for label in classes:
        mask = labels == label
        coverage.append(
            float(np.mean([label in values for values in np.asarray(prediction_sets, dtype=object)[mask]]))
        )
        set_size.append(float(np.mean(cardinality[mask])))
        singleton.append(float(np.mean(cardinality[mask] == 1)))
        empty.append(float(np.mean(cardinality[mask] == 0)))
    return {
        "coverage": float(np.mean(coverage)),
        "average_set_size": float(np.mean(set_size)),
        "singleton_rate": float(np.mean(singleton)),
        "empty_rate": float(np.mean(empty)),
    }


def reference_rows() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line in REFERENCE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if CONFIG["model"] in row["models"]:
            result[int(row["split_id"])] = row
    expected = CONFIG["folds"] * CONFIG["repeats"]
    if len(result) != expected:
        raise RuntimeError(
            f"expected {expected} author-reference splits for {CONFIG['model']}, "
            f"observed {len(result)}"
        )
    return result


def independent_split_worker(split_id: int) -> dict[str, Any]:
    """Run one frozen author fold with the local ConE calibrator."""

    from sklearn.model_selection import RepeatedStratifiedKFold

    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from MBW_LR import MBW_LR
    from utils import create_cali_dataset

    independent_path = ROOT / "src/iia_benchmark/models/cone_afc.py"
    independent_spec = importlib.util.spec_from_file_location(
        "independent_cone_afc", independent_path
    )
    if independent_spec is None or independent_spec.loader is None:
        raise RuntimeError(f"cannot import {independent_path}")
    independent_module = importlib.util.module_from_spec(independent_spec)
    sys.modules[independent_spec.name] = independent_module
    independent_spec.loader.exec_module(independent_module)
    ConEAFCCalibrator = independent_module.ConEAFCCalibrator

    started = time.perf_counter()
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    y = np.load(CACHE / "y.npy", mmap_mode="r")
    splitter = RepeatedStratifiedKFold(
        n_splits=CONFIG["folds"],
        n_repeats=CONFIG["repeats"],
        random_state=CONFIG["random_seed"],
    )
    train_index = test_index = None
    for index, pair in enumerate(splitter.split(X, y)):
        if index == split_id:
            train_index, test_index = pair
            break
    if train_index is None or test_index is None:
        raise ValueError(f"split {split_id} is unavailable")
    outer_train = np.asarray(train_index)
    test_index = np.asarray(test_index)
    X_train, X_test = X[outer_train], X[test_index]
    y_train, y_test = y[outer_train], y[test_index]
    X_train, y_train, X_cal, y_cal = create_cali_dataset(
        X_train,
        y_train,
        random_state=CONFIG["random_seed"],
        size_cali_class=max(CONFIG["calibration_per_class"]),
        use_size_cali_class=max(CONFIG["calibration_per_class"]),
    )
    classes = np.unique(y)
    model_params = {
        "penalty": "none",
        "fit_intercept": False,
        "solver": "lbfgs",
        "multi_class": "ovr",
        "decision_bounds": True,
        "confidence_interval": 1.96,
    }
    calibration_scores: dict[int, np.ndarray] = {}
    test_scores: dict[int, np.ndarray] = {}
    accuracies: list[float] = []
    for prefix in prefixes():
        model = MBW_LR(model_params)
        model.fit(np.asarray(X_train[:, :, :prefix]), np.asarray(y_train))
        cal_score = np.asarray(model.predict_proba(np.asarray(X_cal[:, :, :prefix])))
        test_score = np.asarray(model.predict_proba(np.asarray(X_test[:, :, :prefix])))
        calibration_scores[prefix] = cal_score
        test_scores[prefix] = test_score
        point = classes[np.argmax(test_score, axis=1)]
        accuracies.append(
            float(np.mean([np.mean(point[y_test == label] == label) for label in classes]))
        )

    grid: dict[str, dict[str, float]] = {}
    for alpha in CONFIG["alpha"]:
        for size in CONFIG["calibration_per_class"]:
            calibration_index = np.concatenate(
                [np.flatnonzero(y_cal == label)[:size] for label in classes]
            )
            selected_scores = {
                prefix: values[calibration_index]
                for prefix, values in calibration_scores.items()
            }
            calibrator = ConEAFCCalibrator(
                error_rate=float(alpha), score_kind="probability"
            ).fit(selected_scores, y_cal[calibration_index], classes)
            prefix_metrics = [
                class_balanced_metrics(
                    calibrator.predict_sets(test_scores[prefix], prefix),
                    y_test,
                    classes,
                )
                for prefix in prefixes()
            ]
            key = f"alpha={alpha:.2f}|ncal={size}"
            grid[key] = {
                metric: float(np.mean([row[metric] for row in prefix_metrics]))
                for metric in CONFIG["comparison_metrics"]
            }

    reference = reference_rows()[split_id]
    author = reference["models"][CONFIG["model"]]
    comparisons: list[dict[str, Any]] = []
    for key, observed in grid.items():
        for metric in CONFIG["comparison_metrics"]:
            expected = float(author["grid"][key][metric])
            delta = float(observed[metric] - expected)
            comparisons.append(
                {
                    "condition": key,
                    "metric": metric,
                    "author": expected,
                    "independent": float(observed[metric]),
                    "delta": delta,
                    "within_tolerance": abs(delta)
                    <= CONFIG["exact_absolute_tolerance"],
                }
            )
    accuracy_delta = float(np.mean(accuracies) - author["original_accuracy"])
    return {
        "task_id": f"split={split_id}|model={CONFIG['model']}|layer=independent_cone",
        "split_id": split_id,
        "repeat": split_id // CONFIG["folds"],
        "fold": split_id % CONFIG["folds"],
        "random_seed": CONFIG["random_seed"],
        "scope": CONFIG["validation_scope"],
        "partition_audit": {
            "outer_train_size": int(len(outer_train)),
            "test_size": int(len(test_index)),
            "author_train_after_calibration_size": int(len(y_train)),
            "calibration_size": int(len(y_cal)),
            "outer_train_sha256": sha256_indices(outer_train),
            "test_sha256": sha256_indices(test_index),
        },
        "original_accuracy": float(np.mean(accuracies)),
        "author_original_accuracy": float(author["original_accuracy"]),
        "original_accuracy_delta": accuracy_delta,
        "grid": grid,
        "comparisons": comparisons,
        "rows_within_tolerance": int(sum(row["within_tolerance"] for row in comparisons)),
        "rows_total": len(comparisons),
        "maximum_absolute_delta": float(max(abs(row["delta"]) for row in comparisons)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["split_id"])] = row
    return completed


def write_completed(path: Path, completed: dict[int, dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(completed[key], ensure_ascii=False) + "\n"
        for key in sorted(completed)
    )
    atomic_write_text(path, payload)
    return payload


def summarize(completed: dict[int, dict[str, Any]], payload: str) -> dict[str, Any]:
    expected = CONFIG["folds"] * CONFIG["repeats"]
    rows = [completed[key] for key in sorted(completed) if key < expected]
    comparison_rows = [item for row in rows for item in row["comparisons"]]
    deltas_by_metric = {
        metric: [
            item["delta"]
            for item in comparison_rows
            if item["metric"] == metric
        ]
        for metric in CONFIG["comparison_metrics"]
    }
    return {
        "paper_id": CONFIG["paper_id"],
        "scope": CONFIG["validation_scope"],
        "model": CONFIG["model"],
        "splits_completed": len(rows),
        "splits_required": expected,
        "complete": len(rows) == expected,
        "comparison_rows_within_tolerance": int(
            sum(item["within_tolerance"] for item in comparison_rows)
        ),
        "comparison_rows_total": len(comparison_rows),
        "maximum_absolute_delta": float(
            max((abs(item["delta"]) for item in comparison_rows), default=float("nan"))
        ),
        "accuracy_delta_mean": float(
            np.mean([row["original_accuracy_delta"] for row in rows])
        ),
        "accuracy_delta_max_absolute": float(
            max((abs(row["original_accuracy_delta"]) for row in rows), default=float("nan"))
        ),
        "metric_delta_summary": {
            metric: {
                "mean": float(np.mean(values)),
                "maximum_absolute": float(max(abs(value) for value in values)),
            }
            for metric, values in deltas_by_metric.items()
            if values
        },
        "elapsed_seconds_total": float(sum(row["elapsed_seconds"] for row in rows)),
        "checkpoint_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "author_reference": CONFIG["author_reference"],
    }


def run(workers: int, max_splits: int | None) -> None:
    paper_grid = load_paper_grid_module()
    paper_grid.prepare_cone_cache()
    reference_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    task_path = OUTPUT / "fold_results.jsonl"
    completed = load_completed(task_path)
    required = CONFIG["folds"] * CONFIG["repeats"]
    target = required if max_splits is None else min(required, max_splits)
    requested = list(range(target))
    pending = [split_id for split_id in requested if split_id not in completed]
    print(
        "Experiment purpose: compare the repository-independent ConE conformal "
        "calibrator against frozen author-grid metrics on identical MBW-LR scores, "
        "folds, prefixes, alpha values, and calibration sizes.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(independent_split_worker, split_id): split_id
            for split_id in pending
        }
        for future in as_completed(futures):
            row = future.result()
            completed[int(row["split_id"])] = row
            payload = write_completed(task_path, completed)
            progress = {
                "checkpoint_unit": CONFIG["checkpoint_unit"],
                "splits_completed_total": len(completed),
                "requested_splits_completed": sum(key in completed for key in requested),
                "requested_splits_total": len(requested),
                "checkpoint_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
            atomic_write_text(OUTPUT / "progress.json", json.dumps(progress, indent=2) + "\n")
            print(f"completed independent ConE split {row['split_id'] + 1}/{target}", flush=True)
    payload = write_completed(task_path, completed)
    report = summarize(completed, payload)
    atomic_write_text(
        OUTPUT / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--workers", type=int, default=CONFIG["default_workers"])
    parser.add_argument("--max-splits", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_splits is not None and args.max_splits < 1:
        raise ValueError("max-splits must be positive")
    run(args.workers, args.max_splits)


if __name__ == "__main__":
    main()
