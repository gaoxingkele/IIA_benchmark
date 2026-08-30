#!/usr/bin/env python3
"""Execute paper grids omitted from the published P0 Capsule default scripts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
CONE_EXPORT = ROOT / "data/public_datasets/codeocean/cone_afc_v2/export"
CONE_CACHE = ROOT / "tmp/paper_exact_cache/cone_v2"
CONE_OUTPUT = PROJECT / "run_2/paper_grid"
CASIM_EXPORT = ROOT / "data/public_datasets/codeocean/casim_v1/export"
CASIM_CACHE = ROOT / "tmp/paper_exact_cache/casim_v1"
ALPHAS = (0.01, 0.05, 0.10)
CALIBRATION_SIZES = (22, 102, 2491)
PREFIXES = tuple(range(10, 61))
MODEL_NAMES = ("WDI_1NN", "MBW_LR", "EAC_1NN", "ACM_SVM", "CASIM")


def prepare_casim_cache() -> dict[str, Any]:
    CASIM_CACHE.mkdir(parents=True, exist_ok=True)
    x_path = CASIM_CACHE / "X.npy"
    y_path = CASIM_CACHE / "y.npy"
    if not x_path.is_file() or not y_path.is_file():
        sys.path.insert(0, str(CASIM_EXPORT / "code"))
        from utils import get_X, load_data, load_ground_truth

        source = load_data(str(CASIM_EXPORT / "data") + os.sep)
        y = load_ground_truth(str(CASIM_EXPORT / "data") + os.sep, source)
        X = np.asarray(get_X(source), dtype=np.uint8)
        np.save(x_path, X, allow_pickle=False)
        np.save(y_path, np.asarray(y, dtype=np.int16), allow_pickle=False)
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    metadata = {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_dtype": str(X.dtype),
        "class_counts": {
            str(int(label)): int(np.sum(y == label)) for label in np.unique(y)
        },
        "source_manifest_sha256": "77643dfb40749472f5dfb752611b1db3c462822453e20d6e4669e9cb14f14dd2",
    }
    (CASIM_CACHE / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if metadata["X_shape"] != [310, 76, 4200] or len(metadata["class_counts"]) != 15:
        raise RuntimeError(f"unexpected CASIM v1 cache: {metadata}")
    return metadata


def casim_open_set_worker(task_id: int, repetitions: int) -> dict[str, Any]:
    import warnings

    warnings.filterwarnings("ignore")

    sys.path.insert(0, str(CASIM_EXPORT / "code"))
    from CASIM import CASIM
    from utils import get_train_test

    X = np.load(CASIM_CACHE / "X.npy", mmap_mode="r")
    y = np.load(CASIM_CACHE / "y.npy", mmap_mode="r")
    held_out_classes = tuple(int(value) for value in np.unique(y) if value != -1)
    tasks_per_repetition = len(held_out_classes) * 5
    repetition = task_id // tasks_per_repetition
    within = task_id % tasks_per_repetition
    held_out = held_out_classes[within // 5]
    fold = within % 5
    if repetition >= repetitions:
        raise ValueError("task exceeds requested repetition count")
    # The paper says the complete held-out class is excluded from training and
    # used only for testing across the five folds.  Relabeling that class as
    # -1 and invoking the published open-set splitter implements that wording:
    # every novel sample is assigned to exactly one test fold, while all -1
    # rows are removed from every training fold by the author utility.
    y_open = np.asarray(y).copy()
    y_open[y_open == held_out] = -1
    split = None
    for split_index, values in enumerate(get_train_test(X, y_open, open_set=True)):
        if split_index == fold:
            split = values
            break
    if split is None:
        raise RuntimeError(f"CASIM fold {fold} is unavailable")
    X_train, X_test, y_train, y_test = split
    params = {
        "num_features": 672,
        "n_estimators": 10,
        "n_jobs_multirocket": 1,
        "n_neighbors": 10,
        "novelty_loop": True,
        "random_state": 42 + repetition,
        "extent": 3,
        "alphas": np.logspace(-3, 3, 10),
    }
    model = CASIM(params)
    model.fit(X_train, y_train)
    point, novelty = model.predict_proba(X_test)
    return {
        "task_id": task_id,
        "repetition": repetition,
        "held_out_class": held_out,
        "fold": fold,
        "train_samples": int(len(y_train)),
        "known_test_samples": int(np.sum(y_test != -1)),
        "novel_test_samples": int(np.sum(y_test == -1)),
        "y_true": [int(value) for value in y_test],
        "point_prediction": [int(value) for value in point],
        "novelty_score": [float(value) for value in novelty],
    }


def summarize_casim_open_set(rows: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    threshold_values = np.round(np.arange(0.001, 1.001, 0.001), 3)
    tpr_rows = []
    tnr_rows = []
    bacc_rows = []
    for row in rows:
        y_true = np.asarray(row["y_true"], dtype=int)
        point = np.asarray(row["point_prediction"], dtype=int)
        novelty = np.asarray(row["novelty_score"], dtype=float)
        known = y_true != -1
        novel = ~known
        known_classes = np.unique(y_true[known])
        split_tpr = []
        split_tnr = []
        for threshold in threshold_values:
            # Paper O2.4: accept the point class only when p_out < tau;
            # equality is classified as novel.
            accepted = novelty < threshold
            split_tpr.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                accepted[(y_true == label) & known]
                                & (point[(y_true == label) & known] == label)
                            )
                            for label in known_classes
                        ]
                    )
                )
            )
            split_tnr.append(float(np.mean(novelty[novel] > threshold)))
        tpr = np.asarray(split_tpr)
        tnr = np.asarray(split_tnr)
        tpr_rows.append(tpr)
        tnr_rows.append(tnr)
        bacc_rows.append((tpr + tnr) / 2.0)
    mean_tpr = np.mean(np.vstack(tpr_rows), axis=0)
    mean_tnr = np.mean(np.vstack(tnr_rows), axis=0)
    mean_bacc = np.mean(np.vstack(bacc_rows), axis=0)
    best = int(np.argmax(mean_bacc))
    return {
        "schema_version": 1,
        "paper_id": "faulwasser2024_casim",
        "protocol": "14 leave-one-class-out settings x five stratified folds; the held-out class is relabeled -1 and the published get_train_test(..., open_set=True) splitter assigns every novel sample to exactly one test fold",
        "repetitions": repetitions,
        "train_test_sets": len(rows),
        "thresholds": [float(value) for value in threshold_values],
        "mean_TPR": [float(value) for value in mean_tpr],
        "mean_TNR": [float(value) for value in mean_tnr],
        "mean_balanced_accuracy": [float(value) for value in mean_bacc],
        "maximum": {
            "threshold": float(threshold_values[best]),
            "balanced_accuracy": float(mean_bacc[best]),
            "TPR": float(mean_tpr[best]),
            "TNR": float(mean_tnr[best]),
        },
        "mean_balanced_accuracy_over_thresholds": float(np.mean(mean_bacc)),
        "paper_targets": {
            "maximum_threshold": 0.324,
            "maximum_balanced_accuracy": 0.947,
            "mean_balanced_accuracy_over_thresholds": 0.879,
        },
        "paper_deltas": {
            "maximum_threshold": float(threshold_values[best] - 0.324),
            "maximum_balanced_accuracy": float(mean_bacc[best] - 0.947),
            "mean_balanced_accuracy_over_thresholds": float(np.mean(mean_bacc) - 0.879),
        },
        "construction_status": "paper_text_reconstruction_not_capsule_default",
        "construction_uncertainty": "The Capsule omits the 14-class outer loop. This wrapper supplies only that loop and reuses the published open-set splitter unchanged; the relabel-before-split operation is inferred from the paper wording and remains a protocol uncertainty until confirmed by the authors.",
    }


def run_casim(workers: int, repetitions: int, max_tasks: int | None) -> None:
    metadata = prepare_casim_cache()
    output = PROJECT / f"run_1/paper_grid/repetitions_{repetitions}"
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "seed_results.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if seed_path.is_file():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["task_id"])] = row
    total = repetitions * 14 * 5
    target_count = total if max_tasks is None else min(total, max_tasks)
    pending = [index for index in range(target_count) if index not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(casim_open_set_worker, task_id, repetitions): task_id
            for task_id in pending
        }
        for future in as_completed(futures):
            row = future.result()
            completed[int(row["task_id"])] = row
            ordered = [completed[index] for index in sorted(completed)]
            seed_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
                encoding="utf-8",
            )
            print(f"completed CASIM open-set task {row['task_id'] + 1}/{target_count}", flush=True)
    rows = [completed[index] for index in range(target_count)]
    report = summarize_casim_open_set(rows, repetitions)
    report["cache"] = metadata
    report["complete_paper_grid"] = target_count == total
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare_cone_cache() -> dict[str, Any]:
    CONE_CACHE.mkdir(parents=True, exist_ok=True)
    x_path = CONE_CACHE / "X.npy"
    y_path = CONE_CACHE / "y.npy"
    if not x_path.is_file() or not y_path.is_file():
        sys.path.insert(0, str(CONE_EXPORT / "code"))
        from utils import load_data

        X, y = load_data(str(CONE_EXPORT / "data") + os.sep)
        np.save(x_path, X, allow_pickle=False)
        np.save(y_path, y, allow_pickle=False)
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    metadata = {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "class_counts": {
            str(int(label)): int(np.sum(y == label)) for label in np.unique(y)
        },
        "source_manifest_sha256": "4a42e6a0ec4759b29486c1c24766c7ee586d6429709241ce170900ece714a1c7",
    }
    (CONE_CACHE / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if metadata["X_shape"] != [18750, 10, 60] or metadata["class_counts"] != {
        str(index): 3750 for index in range(5)
    }:
        raise RuntimeError(f"unexpected ConE v2 cache: {metadata}")
    return metadata


def model_definitions() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sys.path.insert(0, str(CONE_EXPORT / "code"))
    from ACM_SVM import ACM_SVM
    from CASIM import CASIM
    from EAC_1NN import EAC_1NN
    from MBW_LR import MBW_LR
    from WDI_1NN import WDI_1NN

    classes = {
        "WDI_1NN": WDI_1NN,
        "MBW_LR": MBW_LR,
        "EAC_1NN": EAC_1NN,
        "ACM_SVM": ACM_SVM,
        "CASIM": CASIM,
    }
    params = {
        "WDI_1NN": {"template_threshold": 0.5},
        "MBW_LR": {
            "penalty": "none",
            "fit_intercept": False,
            "solver": "lbfgs",
            "multi_class": "ovr",
            "decision_bounds": True,
            "confidence_interval": 1.96,
        },
        "EAC_1NN": {"attenuation_coefficient_per_min": 0.0667},
        "ACM_SVM": {},
        "CASIM": {
            "num_features": 672,
            "n_estimators": 1,
            "n_jobs_multirocket": 1,
            "random_state": 42,
            "alphas": np.logspace(-3, 3, 10),
        },
    }
    cone = {
        "WDI_1NN": {"clf_stepwise": True, "input_type": "DIST"},
        "MBW_LR": {"clf_stepwise": True, "input_type": "PROBA"},
        "EAC_1NN": {"clf_stepwise": True, "input_type": "DIST"},
        "ACM_SVM": {"clf_stepwise": False, "input_type": "PROBA"},
        "CASIM": {"clf_stepwise": True, "input_type": "PROBA"},
    }
    return classes, params, cone


def class_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(y_pred[y_true == label] == label) for label in np.unique(y_true)]
        )
    )


def class_balanced_set_metrics(
    prediction_sets: np.ndarray, y_true: np.ndarray, classes: np.ndarray
) -> tuple[float, float, float, float]:
    coverage = []
    sizes = []
    singleton = []
    empty = []
    cardinality = np.sum(prediction_sets, axis=1)
    for class_index, label in enumerate(classes):
        mask = y_true == label
        coverage.append(float(np.mean(prediction_sets[mask, class_index])))
        sizes.append(float(np.mean(cardinality[mask])))
        singleton.append(float(np.mean(cardinality[mask] == 1)))
        empty.append(float(np.mean(cardinality[mask] == 0)))
    return tuple(float(np.mean(values)) for values in (coverage, sizes, singleton, empty))


def thresholds(
    scores: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    *,
    input_type: str,
    alpha: float,
    calibration_per_class: int,
) -> np.ndarray:
    result = []
    for class_index, label in enumerate(classes):
        positions = np.flatnonzero(y == label)[:calibration_per_class]
        values = scores[positions, class_index]
        if input_type == "PROBA":
            ordered = np.sort(values)[::-1]
            q = math.ceil((1.0 - alpha) * len(ordered))
        else:
            ordered = np.sort(values)
            q = math.floor((1.0 - alpha) * len(ordered))
        result.append(float(ordered[q - 1]))
    return np.asarray(result)


def cone_split_worker(split_id: int, selected_models: tuple[str, ...]) -> dict[str, Any]:
    from sklearn.model_selection import RepeatedStratifiedKFold

    sys.path.insert(0, str(CONE_EXPORT / "code"))
    from ConE_AFC import ConE_AFC
    from utils import create_cali_dataset

    X = np.load(CONE_CACHE / "X.npy", mmap_mode="r")
    y = np.load(CONE_CACHE / "y.npy", mmap_mode="r")
    splits = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=42
    ).split(X, y)
    train_index = test_index = None
    for index, pair in enumerate(splits):
        if index == split_id:
            train_index, test_index = pair
            break
    if train_index is None or test_index is None:
        raise RuntimeError(f"split {split_id} is unavailable")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, y_train, X_cali, y_cali = create_cali_dataset(
        X_train,
        y_train,
        random_state=42,
        size_cali_class=2491,
        use_size_cali_class=2491,
    )
    classes = np.unique(y)
    model_classes, params, cone_params = model_definitions()
    result: dict[str, Any] = {
        "split_id": split_id,
        "repeat": split_id // 5,
        "fold": split_id % 5,
        "partition_sizes": {
            "train": int(len(y_train)),
            "calibration": int(len(y_cali)),
            "test": int(len(y_test)),
        },
        "models": {},
    }
    for name in selected_models:
        settings = cone_params[name]
        model = ConE_AFC(
            model_classes[name],
            clf_params=params[name],
            random_state=42,
            step_list=list(PREFIXES),
            clf_stepwise=settings["clf_stepwise"],
            conformal_stepwise=True,
            input_type=settings["input_type"],
            alpha=0.05,
        )
        model.fit(X_train, X_cali, y_train, y_cali)
        original_accuracy = []
        grid = {
            f"alpha={alpha:.2f}|ncal={size}": {
                "coverage": [],
                "average_set_size": [],
                "singleton_rate": [],
                "empty_rate": [],
            }
            for alpha in ALPHAS
            for size in CALIBRATION_SIZES
        }
        for prefix in PREFIXES:
            cal_scores = model.predict_proba(X_cali[:, :, :prefix], prefix)
            test_scores = model.predict_proba(X_test[:, :, :prefix], prefix)
            if settings["input_type"] == "PROBA":
                point = classes[np.argmax(test_scores, axis=1)]
            else:
                point = classes[np.argmin(test_scores, axis=1)]
            original_accuracy.append(class_balanced_accuracy(y_test, point))
            for alpha in ALPHAS:
                for size in CALIBRATION_SIZES:
                    cutoff = thresholds(
                        cal_scores,
                        y_cali,
                        classes,
                        input_type=settings["input_type"],
                        alpha=alpha,
                        calibration_per_class=size,
                    )
                    if settings["input_type"] == "PROBA":
                        prediction_sets = test_scores >= cutoff[None, :]
                    else:
                        prediction_sets = test_scores <= cutoff[None, :]
                    coverage, set_size, singleton, empty = class_balanced_set_metrics(
                        prediction_sets, y_test, classes
                    )
                    target = grid[f"alpha={alpha:.2f}|ncal={size}"]
                    target["coverage"].append(coverage)
                    target["average_set_size"].append(set_size)
                    target["singleton_rate"].append(singleton)
                    target["empty_rate"].append(empty)
        result["models"][name] = {
            "original_accuracy": float(np.mean(original_accuracy)),
            "grid": {
                key: {metric: float(np.mean(values)) for metric, values in row.items()}
                for key, row in grid.items()
            },
        }
    return result


def summarize_cone(rows: list[dict[str, Any]], selected_models: tuple[str, ...]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "paper_id": "faulwasser2024_cone_afc",
        "protocol": "10 repetitions x stratified 5-fold; all 51 prefixes; 3 alpha x 3 calibration sizes",
        "splits": len(rows),
        "models": {},
    }
    for model in selected_models:
        accuracies = np.asarray([row["models"][model]["original_accuracy"] for row in rows])
        grid_keys = rows[0]["models"][model]["grid"]
        report["models"][model] = {
            "original_accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=0)),
            },
            "grid": {},
        }
        for key in grid_keys:
            report["models"][model]["grid"][key] = {}
            for metric in ("coverage", "average_set_size", "singleton_rate", "empty_rate"):
                values = np.asarray([row["models"][model]["grid"][key][metric] for row in rows])
                report["models"][model]["grid"][key][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                }
    card = json.loads(
        (ROOT / "paper_harness/paper_exact/faulwasser2024_cone_afc.v1.json").read_text(
            encoding="utf-8"
        )
    )
    table_1 = card["reference_tables"]["Table_1_accuracy_and_coverage"]
    table_2 = card["reference_tables"]["Table_2_average_set_size"]
    comparisons = []
    for model in selected_models:
        original = report["models"][model]["original_accuracy"]
        paper_mean, paper_std = table_1[model]["original_accuracy"]
        comparisons.append(
            {
                "item": "Table 1",
                "model": model,
                "metric": "original_accuracy",
                "paper_mean": paper_mean,
                "paper_std": paper_std,
                "observed_mean": original["mean"],
                "observed_std": original["std"],
                "mean_delta": original["mean"] - paper_mean,
                "within_mean_tolerance": abs(original["mean"] - paper_mean) <= 0.02,
            }
        )
        for key, observed in report["models"][model]["grid"].items():
            alpha, size = key.replace("alpha=", "").replace("ncal=", "").split("|")
            paper_key = f"{alpha}/{size}"
            for item, metric, source in (
                ("Table 1", "coverage", table_1[model]["coverage"]),
                ("Table 2", "average_set_size", table_2[model]),
            ):
                paper_mean, paper_std = source[paper_key]
                value = observed[metric]
                comparisons.append(
                    {
                        "item": item,
                        "model": model,
                        "metric": metric,
                        "alpha": float(alpha),
                        "calibration_per_class": int(size),
                        "paper_mean": paper_mean,
                        "paper_std": paper_std,
                        "observed_mean": value["mean"],
                        "observed_std": value["std"],
                        "mean_delta": value["mean"] - paper_mean,
                        "within_mean_tolerance": abs(value["mean"] - paper_mean) <= 0.02,
                    }
                )
    report["paper_comparison"] = comparisons
    report["numeric_rows_within_tolerance"] = sum(
        row["within_mean_tolerance"] for row in comparisons
    )
    report["numeric_rows_total"] = len(comparisons)
    return report


def run_cone(workers: int, max_splits: int | None, selected_models: tuple[str, ...]) -> None:
    metadata = prepare_cone_cache()
    CONE_OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_path = CONE_OUTPUT / "seed_results.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if seed_path.is_file():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if tuple(row["models"]) == selected_models:
                    completed[int(row["split_id"])] = row
    target_count = 50 if max_splits is None else min(50, max_splits)
    pending = [index for index in range(target_count) if index not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(cone_split_worker, split_id, selected_models): split_id
            for split_id in pending
        }
        for future in as_completed(futures):
            row = future.result()
            completed[int(row["split_id"])] = row
            ordered = [completed[index] for index in sorted(completed)]
            seed_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
                encoding="utf-8",
            )
            print(f"completed ConE split {row['split_id'] + 1}/{target_count}", flush=True)
    rows = [completed[index] for index in range(target_count)]
    report = summarize_cone(rows, selected_models)
    report["cache"] = metadata
    report["complete_paper_grid"] = target_count == 50 and set(selected_models) == set(MODEL_NAMES)
    (CONE_OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-casim", "run-casim", "prepare-cone", "run-cone")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.command == "prepare-casim":
        print(json.dumps(prepare_casim_cache(), indent=2))
    elif args.command == "run-casim":
        if args.workers < 1 or args.repetitions < 1:
            raise ValueError("workers and repetitions must be positive")
        run_casim(args.workers, args.repetitions, args.max_tasks)
    elif args.command == "prepare-cone":
        print(json.dumps(prepare_cone_cache(), indent=2))
    else:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        selected = tuple(name for name in MODEL_NAMES if name in set(args.models))
        run_cone(args.workers, args.max_splits, selected)


if __name__ == "__main__":
    main()
