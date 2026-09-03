#!/usr/bin/env python3
"""Audit univariate transfer distributions without tuning on held-out data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.evaluation import (  # noqa: E402
    ApplicabilityThresholds,
    assess_univariate_calibration,
    audit_univariate_partitions,
    binary_alarm_metrics,
)
from iia_benchmark.models import (  # noqa: E402
    AdaptedBookUnivariateSuite,
    AdaptiveUnivariateAlarmRouter,
    BlockCalibratedECDFAlarm,
    EmpiricalCDFAlarm,
    FeatureStabilitySelector,
    SafeRollingECDFAlarm,
)


CONFIG = ROOT / "configs/experiments/univariate_adaptation_benchmark.json"
CHAPTER2_EXPERIMENT = ROOT / "experiments/paper_harness/chapter2_multidataset/experiment.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chapter2_module() -> object:
    specification = importlib.util.spec_from_file_location(
        "chapter2_multidataset_experiment", CHAPTER2_EXPERIMENT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the registered Chapter 2 harness")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def git_provenance(source_paths: list[Path]) -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    return {
        "git_worktree_dirty": bool(status),
        "git_status_porcelain": status,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
    }


def calibration_thresholds(config: dict[str, object]) -> ApplicabilityThresholds:
    values = config["applicability_thresholds"]
    return ApplicabilityThresholds(
        normal_ks_adaptation=float(values["normal_ks_adaptation"]),
        normal_median_shift_sd_adaptation=float(
            values["normal_median_shift_sd_adaptation"]
        ),
        autocorrelation_block_calibration=float(
            values["autocorrelation_block_calibration"]
        ),
        minimum_block_auc=float(values["minimum_block_auc"]),
        minimum_direction_consistency=float(
            values["minimum_direction_consistency"]
        ),
        chronological_blocks=int(values["chronological_blocks"]),
    )


def baseline_records(run_name: str) -> tuple[dict[tuple[str, str], dict[str, object]], Path]:
    path = (
        ROOT
        / "experiments/paper_harness/chapter2_multidataset"
        / run_name
        / "final_info.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        (dataset, row["episode_id"]): row
        for dataset, rows in payload["result"]["datasets"].items()
        for row in rows
    }
    return records, path


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    audit = [row["held_out_posthoc_audit"] for row in rows]
    gates = [row["calibration_applicability"] for row in rows]

    def median(path: tuple[str, ...]) -> float:
        values = []
        for record in audit:
            value: object = record
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.median(values))

    adapters = tuple(rows[0]["initial_adapter_results"])
    book_ids = tuple(rows[0]["adapted_book_suite"])
    router_scored = [row["automatic_router"] for row in rows if row["automatic_router"]["empirical"] is not None]
    return {
        "episodes": len(rows),
        "median_normal_train_evaluation_ks": median(
            ("normal_train_to_evaluation", "ks")
        ),
        "median_abnormal_calibration_evaluation_ks": median(
            ("abnormal_calibration_to_evaluation", "ks")
        ),
        "median_evaluation_auc": median(("evaluation_auc",)),
        "median_normal_evaluation_lag_one": median(
            ("normal_evaluation_temporal", "lag_one_autocorrelation")
        ),
        "median_evaluation_abnormal_prevalence": median(
            ("evaluation_abnormal_prevalence",)
        ),
        "calibration_gate_status_counts": {
            status: sum(gate["status"] == status for gate in gates)
            for status in ("static", "adapt", "reject_univariate")
        },
        "selected_features": sorted({str(row["feature"]) for row in rows}),
        "initial_adapter_metrics": {
            adapter: {
                metric: float(
                    np.mean(
                        [
                            row["initial_adapter_results"][adapter]["empirical"][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in (
                    "false_alarm_rate",
                    "missed_alarm_rate",
                    "average_alarm_delay",
                    "f1",
                )
            }
            for adapter in adapters
        },
        "adapted_book_suite_metrics": {
            algorithm_id: {
                metric: float(
                    np.mean(
                        [
                            row["adapted_book_suite"][algorithm_id]["empirical"][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in (
                    "false_alarm_rate",
                    "missed_alarm_rate",
                    "average_alarm_delay",
                    "f1",
                )
            }
            for algorithm_id in book_ids
        },
        "automatic_router": {
            "status_counts": {
                status: sum(row["automatic_router"]["status"] == status for row in rows)
                for status in ("static", "adapt", "reject_univariate")
            },
            "scored_episodes": len(router_scored),
            "denied_episodes": len(rows) - len(router_scored),
            "scored_mean_metrics": {
                metric: (
                    float(np.mean([row["empirical"][metric] for row in router_scored]))
                    if router_scored
                    else None
                )
                for metric in ("false_alarm_rate", "missed_alarm_rate", "f1")
            },
        },
    }


def alarm_metrics(
    model: object, normal: np.ndarray, abnormal: np.ndarray
) -> dict[str, float]:
    prediction = np.r_[model.predict(normal), model.predict(abnormal)]
    truth = np.r_[
        np.zeros(len(normal), dtype=bool), np.ones(len(abnormal), dtype=bool)
    ]
    return binary_alarm_metrics(truth, prediction)


def block_alarm(
    config: dict[str, object], tail: str
) -> BlockCalibratedECDFAlarm:
    values = config["temporal_adaptation"]
    return BlockCalibratedECDFAlarm(
        tail_probability=0.05,
        tail=tail,
        reference_windows=tuple(int(value) for value in values["reference_windows"]),
        delays=tuple(int(value) for value in values["delays"]),
        validation_fraction=float(values["validation_fraction"]),
        block_size=int(values["block_size"]),
        target_point_false_alarm_rate=float(
            values["target_point_false_alarm_rate"]
        ),
        target_block_alarm_rate=float(values["target_block_alarm_rate"]),
        block_weight=float(values["block_weight"]),
    )


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
    chapter2_config_path = ROOT / config["source_experiment"]
    chapter2_config = json.loads(chapter2_config_path.read_text(encoding="utf-8"))
    chapter2 = load_chapter2_module()
    baselines, baseline_path = baseline_records(run_name)
    policy = calibration_thresholds(config)

    print(
        "This run tests whether the selected univariate alarm feature is "
        "distributionally portable. Calibration gates are computed before and "
        "separately from held-out post-hoc diagnostics."
    )
    started = datetime.now(timezone.utc)
    before = time.perf_counter()
    datasets, input_paths = chapter2.load_episodes(chapter2_config)
    results: dict[str, list[dict[str, object]]] = {}
    for dataset, episodes in datasets.items():
        results[dataset] = []
        for episode_index, episode in enumerate(episodes):
            rng = np.random.default_rng(seed + 1009 * episode_index)
            normal_bootstrap = chapter2.bootstrap(
                np.asarray(episode["normal_train"]), rng
            )
            abnormal_bootstrap = chapter2.bootstrap(
                np.asarray(episode["abnormal_calibration"]), rng
            )
            feature_index, feature_name, direction, selection_shift = chapter2.select_feature(
                normal_bootstrap,
                abnormal_bootstrap,
                tuple(episode["feature_names"]),
            )
            baseline = baselines[(dataset, episode["id"])]
            if baseline["feature"] != feature_name or baseline["direction"] != direction:
                raise RuntimeError(
                    f"frozen baseline mismatch for {dataset}/{episode['id']}"
                )
            threshold = float(
                baseline["algorithms"]["book_2_1_iid_delay_timer"]["parameters"][
                    "threshold"
                ]
            )
            normal_train = np.asarray(episode["normal_train"])[:, feature_index]
            normal_evaluation = np.asarray(episode["normal_evaluation"])[
                :, feature_index
            ]
            abnormal_calibration = np.asarray(episode["abnormal_calibration"])[
                :, feature_index
            ]
            abnormal_evaluation = np.asarray(episode["abnormal_evaluation"])[
                :, feature_index
            ]
            calibration_gate = assess_univariate_calibration(
                normal_train, abnormal_calibration, thresholds=policy
            )
            held_out = audit_univariate_partitions(
                normal_train,
                normal_evaluation,
                abnormal_calibration,
                abnormal_evaluation,
                direction=direction,
                threshold=threshold,
            )
            delay = int(
                baseline["algorithms"]["book_2_1_iid_delay_timer"]["parameters"][
                    "delay"
                ]
            )
            one_sided = EmpiricalCDFAlarm(
                tail_probability=0.05, tail=direction, delay=delay
            ).fit(normal_train)
            two_sided = EmpiricalCDFAlarm(
                tail_probability=0.05, tail="two_sided", delay=delay
            ).fit(normal_train)
            stable_selector = FeatureStabilitySelector(
                chronological_blocks=policy.chronological_blocks
            ).fit(
                np.asarray(episode["normal_train"]),
                np.asarray(episode["abnormal_calibration"]),
                feature_names=tuple(episode["feature_names"]),
            )
            stable_normal_train = stable_selector.transform(
                np.asarray(episode["normal_train"])
            )
            stable_normal_evaluation = stable_selector.transform(
                np.asarray(episode["normal_evaluation"])
            )
            stable_abnormal_evaluation = stable_selector.transform(
                np.asarray(episode["abnormal_evaluation"])
            )
            stable_ecdf = EmpiricalCDFAlarm(
                tail_probability=0.05,
                tail=stable_selector.direction_,
                delay=delay,
            ).fit(stable_normal_train)
            block_one_sided = block_alarm(config, direction).fit(normal_train)
            block_two_sided = block_alarm(config, "two_sided").fit(normal_train)
            temporal = config["temporal_adaptation"]
            rolling_window = min(
                int(temporal["safe_rolling_reference_window"]), len(normal_train)
            )
            safe_rolling = SafeRollingECDFAlarm(
                tail_probability=0.05,
                tail="two_sided",
                delay=delay,
                reference_window=rolling_window,
                update_guard_score=float(
                    temporal["safe_rolling_update_guard_score"]
                ),
            ).fit(normal_train)
            initial_adapters = {
                "ecdf_one_sided": {
                    "feature": feature_name,
                    "direction": direction,
                    "empirical": alarm_metrics(
                        one_sided, normal_evaluation, abnormal_evaluation
                    ),
                },
                "ecdf_two_sided": {
                    "feature": feature_name,
                    "direction": "two_sided",
                    "empirical": alarm_metrics(
                        two_sided, normal_evaluation, abnormal_evaluation
                    ),
                },
                "stable_feature_ecdf": {
                    "feature": stable_selector.feature_name_,
                    "direction": stable_selector.direction_,
                    "feature_diagnostics": [
                        item.as_dict() for item in stable_selector.diagnostics_
                    ],
                    "empirical": alarm_metrics(
                        stable_ecdf,
                        stable_normal_evaluation,
                        stable_abnormal_evaluation,
                    ),
                },
                "block_recent_one_sided": {
                    "feature": feature_name,
                    "direction": direction,
                    "parameters": {
                        "reference_window": block_one_sided.selected_reference_window_,
                        "delay": block_one_sided.selected_delay_,
                        "calibration_candidates": [
                            {
                                "reference_window": item.reference_window,
                                "delay": item.delay,
                                "point_false_alarm_rate": item.point_false_alarm_rate,
                                "block_alarm_rate": item.block_alarm_rate,
                                "loss": item.loss,
                            }
                            for item in block_one_sided.calibration_candidates_
                        ],
                    },
                    "empirical": alarm_metrics(
                        block_one_sided, normal_evaluation, abnormal_evaluation
                    ),
                },
                "block_recent_two_sided": {
                    "feature": feature_name,
                    "direction": "two_sided",
                    "parameters": {
                        "reference_window": block_two_sided.selected_reference_window_,
                        "delay": block_two_sided.selected_delay_,
                    },
                    "empirical": alarm_metrics(
                        block_two_sided, normal_evaluation, abnormal_evaluation
                    ),
                },
                "safe_rolling_two_sided": {
                    "feature": feature_name,
                    "direction": "two_sided",
                    "parameters": {
                        "reference_window": rolling_window,
                        "delay": delay,
                        "update_guard_score": float(
                            temporal["safe_rolling_update_guard_score"]
                        ),
                    },
                    "empirical": alarm_metrics(
                        safe_rolling, normal_evaluation, abnormal_evaluation
                    ),
                    "evaluation_update_audit": {
                        "updates": safe_rolling.last_update_count_,
                        "frozen": safe_rolling.last_frozen_count_,
                    },
                },
            }
            chapter2_protocol = chapter2_config["protocol"]
            adapted_suite = AdaptedBookUnivariateSuite(
                tail="two_sided",
                threshold_candidates=int(chapter2_protocol["threshold_candidates"]),
                delays=tuple(int(value) for value in chapter2_protocol["delays"]),
                targets=tuple(float(value) for value in chapter2_protocol["targets"]),
                weights=tuple(float(value) for value in chapter2_protocol["weights"]),
                posterior_confidence=float(
                    chapter2_protocol["posterior_confidence"]
                ),
                app_probability_weight=float(
                    chapter2_protocol["app_probability_weight"]
                ),
            ).fit(normal_train, abnormal_calibration)
            suite_normal = adapted_suite.predict(normal_evaluation)
            suite_abnormal = adapted_suite.predict(abnormal_evaluation)
            adapted_book_results = {
                algorithm_id: {
                    "design": adapted_suite.design_summary_[algorithm_id],
                    "empirical": binary_alarm_metrics(
                        np.r_[
                            np.zeros(len(normal_evaluation), dtype=bool),
                            np.ones(len(abnormal_evaluation), dtype=bool),
                        ],
                        np.r_[suite_normal[algorithm_id], suite_abnormal[algorithm_id]],
                    ),
                }
                for algorithm_id in suite_normal
            }
            router = AdaptiveUnivariateAlarmRouter(
                applicability_thresholds=policy,
                reference_windows=tuple(
                    int(value) for value in temporal["reference_windows"]
                ),
                delays=tuple(int(value) for value in temporal["delays"]),
                block_size=int(temporal["block_size"]),
            ).fit(
                np.asarray(episode["normal_train"]),
                np.asarray(episode["abnormal_calibration"]),
                feature_names=tuple(episode["feature_names"]),
            )
            router_metrics = None
            if router.decision_.status != "reject_univariate":
                router_metrics = binary_alarm_metrics(
                    np.r_[
                        np.zeros(len(episode["normal_evaluation"]), dtype=bool),
                        np.ones(len(episode["abnormal_evaluation"]), dtype=bool),
                    ],
                    np.r_[
                        router.predict(np.asarray(episode["normal_evaluation"])),
                        router.predict(np.asarray(episode["abnormal_evaluation"])),
                    ],
                )
            results[dataset].append(
                {
                    "dataset": dataset,
                    "episode_id": episode["id"],
                    "feature": feature_name,
                    "direction": direction,
                    "selection_standardized_median_shift": selection_shift,
                    "split_policy": episode["split_policy"],
                    "calibration_applicability": calibration_gate.as_dict(),
                    "held_out_posthoc_audit": held_out.as_dict(),
                    "initial_adapter_results": initial_adapters,
                    "adapted_book_suite": adapted_book_results,
                    "automatic_router": {
                        "status": router.decision_.status,
                        "feature": router.decision_.feature_name,
                        "direction": router.decision_.direction,
                        "selected_model": router.decision_.selected_model,
                        "reason": router.decision_.reason,
                        "calibration_applicability": asdict(
                            router.decision_.calibration_applicability
                        ),
                        "empirical": router_metrics,
                    },
                    "frozen_iid_baseline": {
                        "threshold": threshold,
                        "delay": delay,
                        "empirical": baseline["algorithms"][
                            "book_2_1_iid_delay_timer"
                        ]["empirical"],
                    },
                }
            )

    aggregates = {dataset: summarize(rows) for dataset, rows in results.items()}
    beacons = {
        "tep_normal_distribution_stable": bool(
            aggregates["tep_classic"]["median_normal_train_evaluation_ks"] < 0.10
        ),
        "pronto_abnormal_phase_drift": bool(
            aggregates["pronto"]["median_abnormal_calibration_evaluation_ks"]
            > 0.50
        ),
        "skab_normal_baseline_drift": bool(
            aggregates["skab"]["median_normal_train_evaluation_ks"] > 0.40
        ),
        "heldout_not_used_for_routing": True,
    }
    source_paths = [
        CONFIG,
        chapter2_config_path,
        CHAPTER2_EXPERIMENT,
        ROOT / "src/iia_benchmark/evaluation/distribution_audit.py",
        Path(__file__).resolve(),
        baseline_path,
    ]
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "execution_provenance": git_provenance(source_paths),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - before,
        "result": {
            "seed": seed,
            "datasets": results,
            "aggregate_distribution_audit": aggregates,
            "expected_mechanism_beacons": beacons,
            "input_files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in input_paths
            ],
            "reporting_boundary": config["reporting_boundary"],
        },
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
                "aggregates": aggregates,
                "expected_mechanism_beacons": beacons,
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
