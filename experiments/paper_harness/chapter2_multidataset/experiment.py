#!/usr/bin/env python3
"""Run Book Chapter 2 exact numerical and three-dataset validation entries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
from scipy.stats import norm


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import (  # noqa: E402
    load_pronto_merged_csv,
    load_skab_csv,
    load_tep_ascii,
    pronto_normal_train_evaluation_masks,
)
from iia_benchmark.evaluation import binary_alarm_metrics  # noqa: E402
from iia_benchmark.models import (  # noqa: E402
    AlarmOnOffDelay,
    ThresholdDelayDeadband,
    alarm_episode_metrics,
    deadband_index,
    design_deadband_width,
    design_iid_delay_timer,
    design_non_iid_delay_timer,
    iid_delay_timer_performance,
    select_alarm_probability_threshold,
)


CONFIG = ROOT / "configs/experiments/book_chapter2_multidataset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    sources = [
        "src/iia_benchmark/models/univariate_book.py",
        "src/iia_benchmark/models/univariate.py",
        "src/iia_benchmark/data/tep.py",
        "src/iia_benchmark/data/pronto.py",
        "src/iia_benchmark/data/skab.py",
        "experiments/paper_harness/chapter2_multidataset/experiment.py",
        "configs/experiments/book_chapter2_multidataset.json",
    ]
    return {
        "git_worktree_dirty": bool(status),
        "git_status_porcelain": status,
        "source_sha256": {path: sha256_file(ROOT / path) for path in sources},
    }


def segment_bounds(labels: np.ndarray, label: str, occurrence: int) -> tuple[int, int]:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    matches = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if str(labels[start]) == label
    ]
    if occurrence >= len(matches):
        raise ValueError(f"missing {label!r} occurrence {occurrence}")
    return matches[occurrence]


def split_abnormal(values: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    boundary = min(max(1, int(np.floor(len(values) * fraction))), len(values) - 1)
    return values[:boundary], values[boundary:]


def episode_prior(
    normal_train: np.ndarray,
    normal_evaluation: np.ndarray,
    abnormal_calibration: np.ndarray,
    abnormal_evaluation: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, object]:
    blocks = [normal_train, normal_evaluation, abnormal_calibration, abnormal_evaluation]
    return {
        "finite": bool(all(np.isfinite(block).all() for block in blocks)),
        "samples": {
            "normal_train": len(normal_train),
            "normal_evaluation": len(normal_evaluation),
            "abnormal_calibration": len(abnormal_calibration),
            "abnormal_evaluation": len(abnormal_evaluation),
        },
        "features": len(feature_names),
        "constant_features": [
            name
            for index, name in enumerate(feature_names)
            if float(np.std(normal_train[:, index])) <= 1e-12
        ],
        "partition_overlap": 0,
    }


def load_episodes(config: dict[str, object]) -> tuple[dict[str, list[dict[str, object]]], list[Path]]:
    fraction = float(config["protocol"]["abnormal_calibration_fraction"])
    datasets: dict[str, list[dict[str, object]]] = {}
    paths: list[Path] = []

    tep = config["datasets"]["tep_classic"]
    tep_normal_train_path = ROOT / tep["normal_train"]
    tep_normal_eval_path = ROOT / tep["normal_evaluation"]
    tep_normal_train = load_tep_ascii(tep_normal_train_path)
    tep_normal_eval = load_tep_ascii(tep_normal_eval_path)
    tep_episodes = []
    for relative in tep["fault_runs"]:
        path = ROOT / relative
        run = load_tep_ascii(path, fault_start=int(tep["fault_start"]))
        abnormal_cal, abnormal_eval = split_abnormal(
            run.values[int(tep["fault_start"]) :], fraction
        )
        prior = episode_prior(
            tep_normal_train.values,
            tep_normal_eval.values,
            abnormal_cal,
            abnormal_eval,
            run.feature_names,
        )
        tep_episodes.append(
            {
                "id": path.stem,
                "feature_names": run.feature_names,
                "normal_train": tep_normal_train.values,
                "normal_evaluation": tep_normal_eval.values,
                "abnormal_calibration": abnormal_cal,
                "abnormal_evaluation": abnormal_eval,
                "prior": prior,
                "split_policy": "separate normal train/evaluation files; fault run split chronologically after sample 160",
            }
        )
        paths.append(path)
    paths.extend([tep_normal_train_path, tep_normal_eval_path])
    datasets["tep_classic"] = tep_episodes

    pronto = config["datasets"]["pronto"]
    pronto_paths = [ROOT / relative for relative in pronto["paths"]]
    pronto_runs = [load_pronto_merged_csv(path) for path in pronto_paths]
    pronto_episodes = []
    for selected in pronto["selected_fault_segments"]:
        run_index = int(selected["path_index"])
        run = pronto_runs[run_index]
        normal_train_mask, normal_eval_mask = pronto_normal_train_evaluation_masks(
            run.labels,
            train_fraction=float(config["protocol"]["normal_train_fraction"]),
            purge_samples=int(config["protocol"]["normal_purge_samples"]),
        )
        start, stop = segment_bounds(
            run.labels, str(selected["label"]), int(selected["occurrence"])
        )
        abnormal_cal, abnormal_eval = split_abnormal(run.process_values[start:stop], fraction)
        normal_train = run.process_values[normal_train_mask]
        normal_eval = run.process_values[normal_eval_mask & (run.labels == "Normal")]
        prior = episode_prior(
            normal_train,
            normal_eval,
            abnormal_cal,
            abnormal_eval,
            run.process_names,
        )
        pronto_episodes.append(
            {
                "id": f"{run.run_id}_{selected['label'].replace(' ', '_')}_{selected['occurrence']}",
                "feature_names": run.process_names,
                "normal_train": normal_train,
                "normal_evaluation": normal_eval,
                "abnormal_calibration": abnormal_cal,
                "abnormal_evaluation": abnormal_eval,
                "prior": prior,
                "split_policy": "purged chronological normal split plus chronological 40/60 fault-segment split",
            }
        )
    paths.extend(pronto_paths)
    datasets["pronto"] = pronto_episodes

    skab = config["datasets"]["skab"]
    skab_normal_path = ROOT / skab["normal_train"]
    skab_normal = load_skab_csv(skab_normal_path)
    normal_boundary = int(np.floor(len(skab_normal.values) * float(config["protocol"]["normal_train_fraction"])))
    skab_episodes = []
    for relative in skab["fault_runs"]:
        path = ROOT / relative
        run = load_skab_csv(path)
        abnormal_values = run.values[run.abnormal]
        abnormal_cal, abnormal_eval = split_abnormal(abnormal_values, fraction)
        prior = episode_prior(
            skab_normal.values[:normal_boundary],
            skab_normal.values[normal_boundary:],
            abnormal_cal,
            abnormal_eval,
            run.feature_names,
        )
        skab_episodes.append(
            {
                "id": f"{path.parent.name}_{path.stem}",
                "feature_names": run.feature_names,
                "normal_train": skab_normal.values[:normal_boundary],
                "normal_evaluation": skab_normal.values[normal_boundary:],
                "abnormal_calibration": abnormal_cal,
                "abnormal_evaluation": abnormal_eval,
                "prior": prior,
                "split_policy": "anomaly-free file chronological 50/50 split; anomaly samples chronological 40/60 split",
            }
        )
        paths.append(path)
    paths.append(skab_normal_path)
    datasets["skab"] = skab_episodes
    return datasets, sorted(set(paths))


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, len(values), len(values))]


def select_feature(
    normal: np.ndarray,
    abnormal: np.ndarray,
    names: tuple[str, ...],
) -> tuple[int, str, str, float]:
    scale = np.std(normal, axis=0, ddof=1)
    valid = scale > 1e-12
    shifts = np.full(normal.shape[1], -np.inf, dtype=float)
    raw = np.median(abnormal, axis=0) - np.median(normal, axis=0)
    shifts[valid] = np.abs(raw[valid]) / scale[valid]
    index = int(np.argmax(shifts))
    if not np.isfinite(shifts[index]):
        raise ValueError("no nonconstant feature is available")
    direction = "high" if raw[index] >= 0 else "low"
    return index, names[index], direction, float(shifts[index])


def threshold_grid(normal: np.ndarray, abnormal: np.ndarray, count: int) -> np.ndarray:
    combined = np.r_[normal, abnormal]
    grid = np.unique(np.quantile(combined, np.linspace(0.15, 0.85, count)))
    if len(grid) < 2:
        raise ValueError("threshold grid is degenerate")
    return grid


def evaluate(model: object, normal: np.ndarray, abnormal: np.ndarray) -> dict[str, float]:
    values = np.r_[normal, abnormal]
    truth = np.r_[np.zeros(len(normal), dtype=bool), np.ones(len(abnormal), dtype=bool)]
    return binary_alarm_metrics(truth, model.predict(values))


def evaluate_episode(
    episode: dict[str, object],
    dataset: str,
    config: dict[str, object],
    seed: int,
    episode_index: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * episode_index)
    normal_matrix = bootstrap(np.asarray(episode["normal_train"]), rng)
    abnormal_matrix = bootstrap(np.asarray(episode["abnormal_calibration"]), rng)
    names = tuple(episode["feature_names"])
    feature, feature_name, direction, shift = select_feature(normal_matrix, abnormal_matrix, names)
    normal = normal_matrix[:, feature]
    abnormal = abnormal_matrix[:, feature]
    normal_eval = np.asarray(episode["normal_evaluation"])[:, feature]
    abnormal_eval = np.asarray(episode["abnormal_evaluation"])[:, feature]
    protocol = config["protocol"]
    thresholds = threshold_grid(normal, abnormal, int(protocol["threshold_candidates"]))
    delays = [int(value) for value in protocol["delays"]]
    targets = tuple(float(value) for value in protocol["targets"])
    weights = tuple(float(value) for value in protocol["weights"])

    iid = design_iid_delay_timer(
        normal,
        abnormal,
        thresholds=thresholds,
        delays=delays,
        direction=direction,
        targets=targets,
        weights=weights,
    )
    iid_metrics = evaluate(
        AlarmOnOffDelay(iid.threshold, direction, iid.delay),
        normal_eval,
        abnormal_eval,
    )
    iid_predicted = asdict(iid.performance)
    iid_gap = {
        key: float(abs(iid_metrics[metric] - iid_predicted[key]))
        for key, metric in (
            ("false_alarm_rate", "false_alarm_rate"),
            ("missed_alarm_rate", "missed_alarm_rate"),
            ("average_alarm_delay", "average_alarm_delay"),
        )
    }

    non_iid = design_non_iid_delay_timer(
        normal,
        abnormal,
        thresholds=thresholds,
        delays=delays,
        direction=direction,
        target_far=targets[0],
        target_mar=targets[1],
        far_weight=weights[0] / max(weights[0] + weights[1], 1e-12),
        confidence=float(protocol["posterior_confidence"]),
    )
    non_iid_metrics = evaluate(
        AlarmOnOffDelay(non_iid.threshold, direction, non_iid.delay),
        normal_eval,
        abnormal_eval,
    )

    base_threshold = float(np.quantile(normal, 0.95 if direction == "high" else 0.05))
    episodes = alarm_episode_metrics(normal, base_threshold, direction)
    index = deadband_index(normal, base_threshold, direction=direction)
    maximum_width = max(float(np.max(episodes.deviations)), np.finfo(float).eps)
    deadband = design_deadband_width(
        episodes.deviations,
        maximum_width=maximum_width,
        target_remaining_probability=targets[0],
        confidence=float(protocol["posterior_confidence"]),
    )
    activation_threshold = (
        base_threshold + deadband.width
        if direction == "high"
        else base_threshold - deadband.width
    )
    deadband_metrics = evaluate(
        ThresholdDelayDeadband(
            activation_threshold, direction, delay=1, deadband=deadband.width
        ),
        normal_eval,
        abnormal_eval,
    )

    app_samples = np.r_[normal, abnormal]
    minimum_state_samples = max(20, len(app_samples) // 20)
    app = select_alarm_probability_threshold(
        app_samples,
        thresholds,
        minimum_state_samples=minimum_state_samples,
        probability_weight=float(protocol["app_probability_weight"]),
    )
    app_metrics = evaluate(
        ThresholdDelayDeadband(app.threshold, direction), normal_eval, abnormal_eval
    )

    algorithms = {
        "book_2_1_iid_delay_timer": {
            "parameters": {"threshold": iid.threshold, "delay": iid.delay, "direction": direction},
            "design_loss": iid.loss,
            "predicted": iid_predicted,
            "empirical": iid_metrics,
            "absolute_prediction_gap": iid_gap,
        },
        "book_2_2_non_iid_delay_timer": {
            "parameters": {"threshold": non_iid.threshold, "delay": non_iid.delay, "direction": direction},
            "design_loss": non_iid.loss,
            "posterior_far": asdict(non_iid.false_alarm),
            "posterior_mar": asdict(non_iid.missed_alarm),
            "normal_alarm_runs": non_iid.normal_alarm_runs,
            "abnormal_no_alarm_runs": non_iid.abnormal_no_alarm_runs,
            "zero_event_fallback": non_iid.zero_event_fallback,
            "empirical": non_iid_metrics,
        },
        "book_2_3_non_iid_deadband": {
            "parameters": {
                "base_threshold": base_threshold,
                "activation_threshold": activation_threshold,
                "width": deadband.width,
                "direction": direction,
            },
            "index": asdict(index),
            "posterior": asdict(deadband.remaining_probability),
            "empirical": deadband_metrics,
        },
        "book_2_4_alarm_probability_plot": {
            "parameters": {
                "threshold": app.threshold,
                "direction": direction,
                "minimum_state_samples": minimum_state_samples,
                "states": int(len(app.plot.transition_matrix)),
            },
            "score": app.score,
            "empirical": app_metrics,
        },
    }
    for result in algorithms.values():
        metrics = result["empirical"]
        result["execution_passed"] = bool(
            np.isfinite(list(metrics.values())).all()
            and 0.0 <= metrics["false_alarm_rate"] <= 1.0
            and 0.0 <= metrics["missed_alarm_rate"] <= 1.0
        )
    algorithms["book_2_1_iid_delay_timer"]["activation_passed"] = algorithms[
        "book_2_1_iid_delay_timer"
    ]["execution_passed"]
    algorithms["book_2_2_non_iid_delay_timer"]["activation_passed"] = bool(
        algorithms["book_2_2_non_iid_delay_timer"]["execution_passed"]
        and not non_iid.zero_event_fallback
    )
    algorithms["book_2_3_non_iid_deadband"]["activation_passed"] = bool(
        algorithms["book_2_3_non_iid_deadband"]["execution_passed"]
        and index.suitable
    )
    algorithms["book_2_4_alarm_probability_plot"]["activation_passed"] = bool(
        algorithms["book_2_4_alarm_probability_plot"]["execution_passed"]
        and len(app.plot.transition_matrix) >= 2
    )
    return {
        "dataset": dataset,
        "episode_id": episode["id"],
        "feature": feature_name,
        "direction": direction,
        "calibration_standardized_median_shift": shift,
        "prior": episode["prior"],
        "split_policy": episode["split_policy"],
        "algorithms": algorithms,
    }


def simulate_delay_states(exceed: np.ndarray, delay: int) -> np.ndarray:
    active = np.zeros(exceed.shape[0], dtype=bool)
    counter = np.zeros(exceed.shape[0], dtype=int)
    result = np.zeros_like(exceed, dtype=bool)
    for index in range(exceed.shape[1]):
        current = exceed[:, index]
        evidence_for_change = np.where(active, ~current, current)
        counter = np.where(evidence_for_change, counter + 1, 0)
        change = counter >= delay
        active = np.where(change, ~active, active)
        counter = np.where(change, 0, counter)
        result[:, index] = active
    return result


def xu2012_numeric(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    repetitions, samples, burn_in = 500, 1000, 1000
    normal = rng.normal(3.0, 1.0, (repetitions, samples + burn_in))
    abnormal = rng.normal(5.0, 1.0, (repetitions, samples + burn_in))
    q1 = float(norm.sf(1.0))
    p1 = float(norm.sf(-1.0))
    examples = {}
    reported = {
        1: {"far": 0.1589, "mar": 0.1589, "aad": 0.1875},
        3: {"far": 0.0144, "mar": 0.0143, "aad": 3.2805},
    }
    for delay in (1, 3):
        normal_alarm = simulate_delay_states(normal >= 4.0, delay)[:, burn_in:]
        abnormal_alarm = simulate_delay_states(abnormal >= 4.0, delay)[:, burn_in:]
        fresh_abnormal_alarm = simulate_delay_states(
            abnormal[:, :samples] >= 4.0, delay
        )
        first = np.argmax(fresh_abnormal_alarm, axis=1)
        missed = ~np.any(fresh_abnormal_alarm, axis=1)
        first[missed] = samples
        empirical = {
            "far": float(np.mean(normal_alarm)),
            "mar": float(np.mean(~abnormal_alarm)),
            "aad": float(np.mean(first)),
        }
        theory = iid_delay_timer_performance(q1, p1, delay)
        theory_row = {
            "far": theory.false_alarm_rate,
            "mar": theory.missed_alarm_rate,
            "aad": theory.average_alarm_delay,
        }
        examples[f"example_{1 if delay == 1 else 2}"] = {
            "delay": delay,
            "theory": theory_row,
            "paper_monte_carlo_mean": reported[delay],
            "local_monte_carlo_mean": empirical,
            "local_absolute_theory_error": {
                key: abs(empirical[key] - theory_row[key]) for key in empirical
            },
            "local_monte_carlo_passed": bool(
                abs(empirical["far"] - theory_row["far"]) <= 0.005
                and abs(empirical["mar"] - theory_row["mar"]) <= 0.005
                and abs(empirical["aad"] - theory_row["aad"]) <= 0.15
            ),
        }
    table_reference = {
        2: (0.0468, 0.0305, 1.4294),
        3: (0.0116, 0.0060, 2.8988),
        4: (0.0025, 0.0010, 4.5694),
    }
    table_rows = []
    for delay, reference in table_reference.items():
        result = iid_delay_timer_performance(0.1486, 1.0 - 0.1204, delay)
        observed = (
            result.false_alarm_rate,
            result.missed_alarm_rate,
            result.average_alarm_delay,
        )
        table_rows.append(
            {
                "delay": delay,
                "reference": dict(zip(("far", "mar", "aad"), reference, strict=True)),
                "observed": dict(zip(("far", "mar", "aad"), observed, strict=True)),
                "maximum_absolute_error": float(np.max(np.abs(np.asarray(observed) - reference))),
            }
        )
    return {
        "paper": "xu2012_far_mar_aad",
        "items": ["Example 1", "Example 2", "Table VII"],
        "examples": examples,
        "table_vii": table_rows,
        "table_vii_passed_at_1e_4": all(row["maximum_absolute_error"] <= 1e-4 for row in table_rows),
        "examples_passed": all(row["local_monte_carlo_passed"] for row in examples.values()),
        "protocol_fidelity": "P2 equation-defined and published rounded-table reproduction",
    }


def aggregate_episode_results(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    episodes = list(rows)
    algorithm_ids = tuple(episodes[0]["algorithms"])
    result = {}
    for algorithm_id in algorithm_ids:
        values = [row["algorithms"][algorithm_id] for row in episodes]
        result[algorithm_id] = {
            "episodes": len(values),
            "execution_rate": float(np.mean([value["execution_passed"] for value in values])),
            "activation_rate": float(np.mean([value["activation_passed"] for value in values])),
            "mean_false_alarm_rate": float(np.mean([value["empirical"]["false_alarm_rate"] for value in values])),
            "mean_missed_alarm_rate": float(np.mean([value["empirical"]["missed_alarm_rate"] for value in values])),
            "mean_average_alarm_delay": float(np.mean([value["empirical"]["average_alarm_delay"] for value in values])),
            "mean_f1": float(np.mean([value["empirical"]["f1"] for value in values])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    run_name = out_dir.name
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if run_name not in config["runs"]:
        raise ValueError(f"out_dir basename must be one of {sorted(config['runs'])}")
    seed = int(config["runs"][run_name]["seed"])
    started = datetime.now(timezone.utc)
    execution_provenance = provenance()
    before = time.perf_counter()
    datasets, paths = load_episodes(config)
    episode_results = {
        dataset: [
            evaluate_episode(episode, dataset, config, seed, index)
            for index, episode in enumerate(episodes)
        ]
        for dataset, episodes in datasets.items()
    }
    aggregates = {
        dataset: aggregate_episode_results(rows) for dataset, rows in episode_results.items()
    }
    prior_passed = all(
        row["prior"]["finite"] and not row["prior"]["constant_features"]
        for rows in episode_results.values()
        for row in rows
    )
    result = {
        "seed": seed,
        "exact_reproduction": xu2012_numeric(seed),
        "datasets": episode_results,
        "aggregate_metrics": aggregates,
        "prior_gate": {
            "passed": prior_passed,
            "episodes": sum(len(rows) for rows in episode_results.values()),
            "fcc_alarm_denied": True,
            "fcc_reason": config["datasets"]["fcc_alarm"]["status"],
        },
        "activation": {
            algorithm_id: bool(
                all(
                    aggregate[algorithm_id]["activation_rate"] == 1.0
                    for aggregate in aggregates.values()
                )
            )
            for algorithm_id in config["validated_algorithms"]
        },
        "input_files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
        "reporting_boundary": config["reporting_boundary"],
    }
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "execution_provenance": execution_provenance,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - before,
        "result": result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_name": run_name,
                "seed": seed,
                "exact_table_passed": result["exact_reproduction"]["table_vii_passed_at_1e_4"],
                "prior_gate": result["prior_gate"],
                "activation": result["activation"],
                "aggregate_metrics": result["aggregate_metrics"],
                "wall_clock_seconds": payload["wall_clock_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
