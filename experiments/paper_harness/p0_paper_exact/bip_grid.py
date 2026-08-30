#!/usr/bin/env python3
"""Run resumable BiP-AFC paper grids without changing the author models."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
CONFIG_PATH = ROOT / "configs/experiments/p0_bip_paper_grid.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()
EXPORT = ROOT / CONFIG["capsule_export"]
CODE = EXPORT / "code"
DATA = EXPORT / "data"
CACHE = ROOT / CONFIG["runtime_cache"]
OUTPUT = ROOT / CONFIG["output_root"]
CARD_PATH = ROOT / CONFIG["protocol_card"]
RUNTIME_CACHE = ROOT / "tmp/paper_exact_runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(RUNTIME_CACHE / "pycache")
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE / "numba"))


def atomic_write_text(path: Path, payload: str) -> None:
    """Replace a UTF-8 file atomically without Windows newline translation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    os.replace(temporary, path)


def sha256_indices(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _author_data_loader(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from utils import load_data

    source = DATA / CONFIG["datasets"][dataset_name]["data_subdirectory"].split("/", 1)[1]
    return load_data(str(source) + os.sep, dataset_name)


def prepare_cache() -> dict[str, Any]:
    """Materialize read-only NumPy caches from the hash-frozen Capsule data."""

    CACHE.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "source_manifest_sha256": CONFIG["source_manifest_sha256"],
        "datasets": {},
    }
    for dataset_name, dataset_cfg in CONFIG["datasets"].items():
        x_path = CACHE / f"{dataset_name}_X.npy"
        y_path = CACHE / f"{dataset_name}_y.npy"
        if not x_path.is_file() or not y_path.is_file():
            X, y = _author_data_loader(dataset_name)
            np.save(x_path, np.asarray(X, dtype=np.uint8), allow_pickle=False)
            np.save(y_path, np.asarray(y, dtype=np.int16), allow_pickle=False)
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        expected = dataset_cfg["expected_shape"]
        if list(X.shape) != expected or list(y.shape) != [expected[0]]:
            raise RuntimeError(
                f"unexpected {dataset_name} cache shapes: X={X.shape}, y={y.shape}"
            )
        class_counts = {
            str(int(label)): int(np.sum(y == label)) for label in np.unique(y)
        }
        if len(class_counts) != 5:
            raise RuntimeError(f"unexpected {dataset_name} classes: {class_counts}")
        metadata["datasets"][dataset_name] = {
            "X_shape": list(X.shape),
            "X_dtype": str(X.dtype),
            "y_shape": list(y.shape),
            "y_dtype": str(y.dtype),
            "class_counts": class_counts,
        }
    atomic_write_text(CACHE / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    return metadata


def configured_prefixes() -> tuple[int, ...]:
    values = CONFIG["prefix_minutes"]
    return tuple(range(values["start"], values["stop"] + 1, values["step"]))


def configured_alphas() -> np.ndarray:
    values = CONFIG["jackknife_plus_alpha"]
    stop = values["stop"] + values["step"] / 2
    return np.round(np.arange(values["start"], stop, values["step"]), 2)


def outer_fold_indices(dataset_name: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = np.load(CACHE / f"{dataset_name}_y.npy", mmap_mode="r")
    splitter = RepeatedStratifiedKFold(
        n_splits=CONFIG["folds"],
        n_repeats=CONFIG["repeats"],
        random_state=CONFIG["random_seed"],
    )
    for fold_id, (train_index, test_index) in enumerate(
        splitter.split(np.zeros((len(y), 1), dtype=np.uint8), y)
    ):
        if fold_id == fold:
            return np.asarray(train_index), np.asarray(test_index)
    raise ValueError(f"fold {fold} is outside the configured grid")


def partition_fold(
    dataset_name: str, fold: int, split_lane: str
) -> dict[str, np.ndarray]:
    """Reproduce the author split or the paper-text disjoint correction."""

    if split_lane not in CONFIG["split_lanes"]:
        raise ValueError(f"unknown split lane: {split_lane}")
    y = np.load(CACHE / f"{dataset_name}_y.npy", mmap_mode="r")
    outer_train, test = outer_fold_indices(dataset_name, fold)
    return partition_from_outer(
        np.asarray(y),
        outer_train,
        test,
        CONFIG["datasets"][dataset_name],
        split_lane,
    )


def partition_from_outer(
    y: np.ndarray,
    outer_train: np.ndarray,
    test: np.ndarray,
    sizes: dict[str, Any],
    split_lane: str,
) -> dict[str, np.ndarray]:
    """Apply the documented within-fold split to explicit global indices."""

    if split_lane not in CONFIG["split_lanes"]:
        raise ValueError(f"unknown split lane: {split_lane}")
    y_pool = np.asarray(y[outer_train])
    n_cal = int(sizes["cp_calibration_per_class"])
    n_train = int(sizes["afc_train_per_class"])
    n_rf = int(sizes["bip_regressor_train_per_class"])
    calibration: list[np.ndarray] = []
    training: list[np.ndarray] = []
    regressor: list[np.ndarray] = []
    for label in np.unique(y[test]):
        local = np.where(y_pool == label)[0]
        # The Capsule deliberately reinitializes the same seed for every class.
        prng = np.random.RandomState(CONFIG["random_seed"])
        prng.shuffle(local)
        calibration.append(outer_train[local[:n_cal]])
        training.append(outer_train[local[n_cal : n_cal + n_train]])
        if split_lane == "author_overlap":
            regressor.append(outer_train[local[:n_rf]])
        else:
            start = n_cal + n_train
            regressor.append(outer_train[local[start : start + n_rf]])
    result = {
        "afc_train": np.concatenate(training),
        "cp_calibration": np.concatenate(calibration),
        "bip_regressor_train": np.concatenate(regressor),
        "test": np.asarray(test),
    }
    expected = {
        "afc_train": n_train * 5,
        "cp_calibration": n_cal * 5,
        "bip_regressor_train": n_rf * 5,
    }
    for name, count in expected.items():
        if len(result[name]) != count:
            raise RuntimeError(
                f"{split_lane} {name}: "
                f"expected {count}, observed {len(result[name])}"
            )
    return result


def partition_audit(partitions: dict[str, np.ndarray]) -> dict[str, Any]:
    names = tuple(partitions)
    overlaps = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}|{right}"] = int(
                np.intersect1d(partitions[left], partitions[right]).size
            )
    return {
        "sizes": {name: int(len(values)) for name, values in partitions.items()},
        "index_sha256": {
            name: sha256_indices(values) for name, values in partitions.items()
        },
        "pairwise_overlap": overlaps,
    }


def _model_parameters(model_name: str) -> dict[str, Any]:
    parameters = dict(CONFIG["models"][model_name]["parameters"])
    logspace = parameters.pop("ridge_alphas_logspace", None)
    if logspace is not None:
        parameters["alphas"] = np.logspace(*logspace)
    return parameters


def _model_classes(model_name: str) -> tuple[type, type]:
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    bip_class = getattr(importlib.import_module("BiP_AFC"), "BiP_AFC")
    model_cfg = CONFIG["models"][model_name]
    model_class = getattr(
        importlib.import_module(model_cfg["module"]), model_cfg["class"]
    )
    return bip_class, model_class


def _bifurcation_trajectories(
    conformal_sets: dict[int, np.ndarray], prefixes: tuple[int, ...], sample_count: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    future: dict[int, np.ndarray] = {}
    previous: dict[int, np.ndarray] = {}
    for prefix_index, prefix in enumerate(prefixes[:-1]):
        future_values: list[int | None] = []
        previous_values: list[int] = []
        for sample in range(sample_count):
            current = conformal_sets[prefix][sample]
            next_reduction = None
            for later in prefixes[prefix_index + 1 :]:
                next_set = conformal_sets[later][sample]
                if 0 < len(next_set) < len(current):
                    next_reduction = later - prefix
                    break
            last_reduction = 0
            for earlier in prefixes[:prefix_index][::-1]:
                if len(conformal_sets[earlier][sample]) > len(current):
                    last_reduction = abs(prefix - earlier)
                    break
            future_values.append(next_reduction)
            previous_values.append(last_reduction)
        future[prefix] = np.asarray(future_values, dtype=object)
        previous[prefix] = np.asarray(previous_values, dtype=int)
    return future, previous


def _test_conformal_trajectory(
    classifier: Any, X_test: np.ndarray, prefixes: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    conformal_sets: dict[int, np.ndarray] = {}
    conformal_vectors: dict[int, np.ndarray] = {}
    probabilities: dict[int, np.ndarray] = {}
    set_counter = None
    for prefix_index, prefix in enumerate(prefixes):
        previous_set = (
            [set() for _ in range(len(X_test))]
            if prefix_index == 0
            else conformal_sets[prefixes[prefix_index - 1]]
        )
        set_counter, sets, vectors, probas = classifier.predict_conformal(
            X_test[:, :, :prefix],
            step=prefix,
            set_counter=set_counter,
            delay_timer=CONFIG["delay_timer_minutes"],
            previous_set=previous_set,
        )
        conformal_sets[prefix] = sets
        conformal_vectors[prefix] = vectors
        probabilities[prefix] = probas
    return conformal_sets, conformal_vectors, probabilities


def bip_fold_worker(
    split_lane: str, dataset_name: str, model_name: str, fold: int
) -> dict[str, Any]:
    """Run one independently checkpointable author dataset/model/fold task."""

    import warnings

    warnings.filterwarnings("ignore")
    started = time.perf_counter()
    X = np.load(CACHE / f"{dataset_name}_X.npy", mmap_mode="r")
    y = np.load(CACHE / f"{dataset_name}_y.npy", mmap_mode="r")
    partitions = partition_fold(dataset_name, fold, split_lane)
    X_train = np.asarray(X[partitions["afc_train"]])
    y_train = np.asarray(y[partitions["afc_train"]])
    X_cal = np.asarray(X[partitions["cp_calibration"]])
    y_cal = np.asarray(y[partitions["cp_calibration"]])
    X_rf = np.asarray(X[partitions["bip_regressor_train"]])
    X_test = np.asarray(X[partitions["test"]])

    bip_class, model_class = _model_classes(model_name)
    model_cfg = CONFIG["models"][model_name]
    prefixes = configured_prefixes()
    classifier = bip_class(
        classifier=model_class,
        clf_params=_model_parameters(model_name),
        random_state=CONFIG["random_seed"],
        step_list=list(prefixes),
        clf_stepwise=model_cfg["clf_stepwise"],
        input_type=model_cfg["input_type"],
        alpha=CONFIG["conformal_alpha"],
        delay_timer=CONFIG["delay_timer_minutes"],
    )
    classifier.fit(X_train, X_cal, X_rf, y_train, y_cal)
    train_bifurcations = int(
        sum(
            np.sum(classifier.bifurcation_times[prefix] == 1)
            for prefix in prefixes[:-1]
        )
    )

    sets, vectors, probabilities = _test_conformal_trajectory(
        classifier, X_test, prefixes
    )
    future, previous = _bifurcation_trajectories(sets, prefixes, len(X_test))
    test_bifurcations = int(
        sum(np.sum(future[prefix] == 1) for prefix in prefixes[:-1])
    )

    alpha_metrics: list[dict[str, Any]] = []
    for alpha in configured_alphas():
        truth: list[float] = []
        point: list[float] = []
        intervals: list[np.ndarray] = []
        for prefix in prefixes[:-1]:
            prediction, prediction_interval = classifier.predict_rf(
                X_test[:, :, :prefix],
                probabilities[prefix],
                vectors[prefix],
                previous[prefix],
                step=prefix,
                alpha=float(alpha),
                length=X_test.shape[2],
            )
            prediction_interval = np.asarray(prediction_interval)
            if prediction_interval.ndim == 3:
                prediction_interval = prediction_interval[:, :, 0]
            for sample, target in enumerate(future[prefix]):
                if target is not None:
                    truth.append(float(target))
                    point.append(float(prediction[sample]))
                    intervals.append(np.asarray(prediction_interval[sample], dtype=float))
        truth_values = np.asarray(truth)
        point_values = np.asarray(point)
        interval_values = np.asarray(intervals)
        if not len(truth_values):
            raise RuntimeError(
                f"no test bifurcation targets for {split_lane}/{dataset_name}/"
                f"{model_name}/fold={fold}"
            )
        absolute_error = np.abs(truth_values - point_values)
        width = interval_values[:, 1] - interval_values[:, 0]
        covered = (truth_values >= interval_values[:, 0]) & (
            truth_values <= interval_values[:, 1]
        )
        alpha_metrics.append(
            {
                "alpha": float(alpha),
                "target_count": int(len(truth_values)),
                "point_mae": float(np.mean(absolute_error)),
                "point_absolute_error_std": float(np.std(absolute_error)),
                "coverage": float(np.mean(covered)),
                "average_interval_width": float(np.mean(width)),
                "interval_width_std": float(np.std(width)),
            }
        )

    return {
        "task_id": (
            f"lane={split_lane}|dataset={dataset_name}|model={model_name}|fold={fold}"
        ),
        "split_lane": split_lane,
        "dataset": dataset_name,
        "model": model_name,
        "fold": fold,
        "random_seed": CONFIG["random_seed"],
        "partition_audit": partition_audit(partitions),
        "train_bifurcations": train_bifurcations,
        "test_bifurcations": test_bifurcations,
        "alpha_metrics": alpha_metrics,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def task_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        row["split_lane"],
        row["dataset"],
        row["model"],
        int(row["fold"]),
    )


def load_tasks(path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    completed = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[task_key(row)] = row
    return completed


def write_tasks(
    path: Path, completed: dict[tuple[str, str, str, int], dict[str, Any]]
) -> str:
    payload = "".join(
        json.dumps(completed[key], ensure_ascii=False) + "\n"
        for key in sorted(completed)
    )
    atomic_write_text(path, payload)
    return payload


def write_progress(
    payload: str,
    completed: dict[tuple[str, str, str, int], dict[str, Any]],
    requested: list[tuple[str, str, str, int]],
) -> None:
    progress = {
        "checkpoint_unit": CONFIG["checkpoint_unit"],
        "tasks_completed_total": len(completed),
        "requested_tasks_completed": sum(task in completed for task in requested),
        "requested_tasks_total": len(requested),
        "checkpoint_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    atomic_write_text(
        OUTPUT / "progress.json",
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
    )


def _sample_std(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["fold"])
    alpha_values = configured_alphas()
    coverage_matrix = np.asarray(
        [[metric["coverage"] for metric in row["alpha_metrics"]] for row in rows]
    )
    point_matrix = np.asarray(
        [[metric["point_mae"] for metric in row["alpha_metrics"]] for row in rows]
    )
    point_std_matrix = np.asarray(
        [
            [metric["point_absolute_error_std"] for metric in row["alpha_metrics"]]
            for row in rows
        ]
    )
    width_matrix = np.asarray(
        [
            [metric["average_interval_width"] for metric in row["alpha_metrics"]]
            for row in rows
        ]
    )
    nominal = 1.0 - alpha_values
    return {
        "folds": len(rows),
        "train_bifurcations": {
            "mean": float(np.mean([row["train_bifurcations"] for row in rows])),
            "std": _sample_std(row["train_bifurcations"] for row in rows),
            "values": [row["train_bifurcations"] for row in rows],
        },
        "test_bifurcations": {
            "mean": float(np.mean([row["test_bifurcations"] for row in rows])),
            "std": _sample_std(row["test_bifurcations"] for row in rows),
            "values": [row["test_bifurcations"] for row in rows],
        },
        "coverage_mae": {
            "mean": float(np.mean(np.abs(coverage_matrix - nominal[None, :]))),
            "paper_std": float(
                np.mean(np.std(coverage_matrix, axis=0, ddof=1))
                if len(rows) > 1
                else 0.0
            ),
            "fold_mae_std": _sample_std(
                np.mean(np.abs(row - nominal)) for row in coverage_matrix
            ),
        },
        "point_mae": {
            "mean": float(np.mean(point_matrix)),
            "paper_std": float(np.mean(point_std_matrix)),
            "fold_mean_std": _sample_std(np.mean(row) for row in point_matrix),
        },
        "interval_width": {
            "mean": float(np.mean(width_matrix)),
            "fold_mean_std": _sample_std(np.mean(row) for row in width_matrix),
        },
        "alpha_curves": [
            {
                "alpha": float(alpha),
                "coverage_mean": float(np.mean(coverage_matrix[:, index])),
                "coverage_std": _sample_std(coverage_matrix[:, index]),
                "point_mae_mean": float(np.mean(point_matrix[:, index])),
                "point_mae_std": _sample_std(point_matrix[:, index]),
                "interval_width_mean": float(np.mean(width_matrix[:, index])),
                "interval_width_std": _sample_std(width_matrix[:, index]),
            }
            for index, alpha in enumerate(alpha_values)
        ],
    }


def compare_paper(groups: dict[str, Any]) -> list[dict[str, Any]]:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    tolerance = card["tolerances"]
    comparisons = []
    author = groups.get("author_overlap", {})
    for target in card["paper_targets"]:
        observed = author.get(target["dataset"], {}).get(target["model"])
        comparison: dict[str, Any] = {
            "item": target["item"],
            "dataset": target["dataset"],
            "model": target["model"],
            "metric": target["metric"],
            "complete": observed is not None and observed["folds"] == CONFIG["folds"],
        }
        if comparison["complete"] and target["metric"] == "bifurcations":
            train = observed["train_bifurcations"]
            test = observed["test_bifurcations"]
            comparison.update(
                {
                    "paper_train_mean": target["train_mean"],
                    "observed_train_mean": train["mean"],
                    "paper_train_std": target["train_std"],
                    "observed_train_std": train["std"],
                    "paper_test_mean": target["test_mean"],
                    "observed_test_mean": test["mean"],
                    "paper_test_std": target["test_std"],
                    "observed_test_std": test["std"],
                    "within_tolerance": (
                        abs(train["mean"] - target["train_mean"])
                        / target["train_mean"]
                        <= tolerance["bifurcation_count_relative"]
                        and abs(test["mean"] - target["test_mean"])
                        / target["test_mean"]
                        <= tolerance["bifurcation_count_relative"]
                    ),
                }
            )
        elif comparison["complete"]:
            coverage = observed["coverage_mae"]
            points = observed["point_mae"]
            comparison.update(
                {
                    "paper_coverage_mean": target["coverage_mean"],
                    "observed_coverage_mean": coverage["mean"],
                    "paper_coverage_std": target["coverage_std"],
                    "observed_coverage_std": coverage["paper_std"],
                    "paper_points_mean": target["points_mean"],
                    "observed_points_mean": points["mean"],
                    "paper_points_std": target["points_std"],
                    "observed_points_std": points["paper_std"],
                    "within_tolerance": (
                        abs(coverage["mean"] - target["coverage_mean"])
                        <= tolerance["MAE_coverage_absolute"]
                        and abs(points["mean"] - target["points_mean"])
                        / target["points_mean"]
                        <= tolerance["MAE_points_relative"]
                    ),
                }
            )
        else:
            comparison["within_tolerance"] = False
        comparisons.append(comparison)
    return comparisons


def summarize(
    completed: dict[tuple[str, str, str, int], dict[str, Any]],
    requested_tasks: list[tuple[str, str, str, int]],
    checkpoint_payload: str,
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for lane in CONFIG["split_lanes"]:
        groups[lane] = {}
        for dataset_name in CONFIG["datasets"]:
            groups[lane][dataset_name] = {}
            for model_name in CONFIG["models"]:
                rows = [
                    completed[(lane, dataset_name, model_name, fold)]
                    for fold in range(CONFIG["folds"])
                    if (lane, dataset_name, model_name, fold) in completed
                ]
                if rows:
                    groups[lane][dataset_name][model_name] = summarize_group(rows)
    comparisons = compare_paper(groups)
    all_tasks = [
        (lane, dataset, model, fold)
        for lane in CONFIG["split_lanes"]
        for dataset in CONFIG["datasets"]
        for model in CONFIG["models"]
        for fold in range(CONFIG["folds"])
    ]
    author_tasks = [task for task in all_tasks if task[0] == "author_overlap"]
    return {
        "paper_id": CONFIG["paper_id"],
        "protocol": {
            "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "card": CONFIG["protocol_card"],
            "checkpoint_unit": CONFIG["checkpoint_unit"],
            "author_code_changed": False,
            "author_overlap_retained": True,
            "paper_disjoint_ablation_definition": CONFIG["split_lanes"][
                "paper_disjoint"
            ]["description"],
        },
        "cache": cache_metadata,
        "groups": groups,
        "paper_comparison": comparisons,
        "numeric_rows_within_tolerance": sum(
            bool(row["within_tolerance"]) for row in comparisons
        ),
        "numeric_rows_total": len(comparisons),
        "tasks_completed": sum(task in completed for task in all_tasks),
        "tasks_required_author_grid": len(author_tasks),
        "tasks_required_with_ablation": len(all_tasks),
        "requested_tasks_completed": sum(task in completed for task in requested_tasks),
        "requested_tasks_total": len(requested_tasks),
        "complete_author_paper_grid": all(task in completed for task in author_tasks),
        "complete_required_ablation": all(task in completed for task in all_tasks),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_payload.encode("utf-8")
        ).hexdigest(),
    }


def run_bip(
    workers: int,
    lanes: tuple[str, ...],
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    max_tasks: int | None,
) -> None:
    cache_metadata = prepare_cache()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    task_path = OUTPUT / "fold_results.jsonl"
    completed = load_tasks(task_path)
    requested = [
        (lane, dataset, model, fold)
        for lane in lanes
        for dataset in datasets
        for model in models
        for fold in range(CONFIG["folds"])
    ]
    if max_tasks is not None:
        requested = requested[:max_tasks]
    pending = [task for task in requested if task not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(bip_fold_worker, *task): task for task in pending
        }
        for future in as_completed(futures):
            row = future.result()
            completed[task_key(row)] = row
            checkpoint_payload = write_tasks(task_path, completed)
            write_progress(checkpoint_payload, completed, requested)
            print(f"completed BiP {row['task_id']}", flush=True)
    checkpoint_payload = write_tasks(task_path, completed)
    write_progress(checkpoint_payload, completed, requested)
    report = summarize(completed, requested, checkpoint_payload, cache_metadata)
    atomic_write_text(
        OUTPUT / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-bip", "run-bip"))
    parser.add_argument(
        "--workers", type=int, default=int(CONFIG.get("default_workers", 4))
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=tuple(CONFIG["split_lanes"]),
        default=["author_overlap"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(CONFIG["datasets"]),
        default=list(CONFIG["datasets"]),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(CONFIG["models"]),
        default=list(CONFIG["models"]),
    )
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("max-tasks must be positive")
    if args.command == "prepare-bip":
        print(json.dumps(prepare_cache(), indent=2))
    else:
        run_bip(
            args.workers,
            tuple(args.lanes),
            tuple(args.datasets),
            tuple(args.models),
            args.max_tasks,
        )


if __name__ == "__main__":
    main()
