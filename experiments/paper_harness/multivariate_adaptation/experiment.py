#!/usr/bin/env python3
"""Run multi-dataset multivariate distribution adaptation experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.paper_harness.chapter3_multidataset.experiment import (  # noqa: E402
    contiguous_fit_block,
    load_episodes,
    select_features,
)
from iia_benchmark.evaluation import (  # noqa: E402
    MultivariateApplicabilityThresholds,
    alarm_event_metrics,
    binary_alarm_metrics,
    block_bootstrap_alarm_metrics,
    multivariate_distribution_shift,
)
from iia_benchmark.models import (  # noqa: E402
    AdaptiveMultivariateAlarmRouter,
    BlockCalibratedRobustMahalanobisAlarm,
    MahalanobisAlarm,
    RobustShrinkageMahalanobisAlarm,
)


CONFIG = ROOT / "configs/experiments/multivariate_adaptation_benchmark.json"
SOURCE_CONFIG = ROOT / "configs/experiments/book_chapter3_multidataset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_prediction(
    normal_alarm: np.ndarray,
    abnormal_alarm: np.ndarray,
    *,
    sample_period_seconds: float,
    block_size: int,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    truth = np.r_[
        np.zeros(len(normal_alarm), dtype=bool),
        np.ones(len(abnormal_alarm), dtype=bool),
    ]
    return {
        "empirical": binary_alarm_metrics(truth, np.r_[normal_alarm, abnormal_alarm]),
        "event_metrics": alarm_event_metrics(
            normal_alarm,
            abnormal_alarm,
            sample_period_seconds=sample_period_seconds,
        ).as_dict(),
        "block_bootstrap": block_bootstrap_alarm_metrics(
            normal_alarm,
            abnormal_alarm,
            block_size=block_size,
            draws=draws,
            confidence=confidence,
            seed=seed,
        ).as_dict(),
    }


def evaluate_episode(
    episode: dict[str, object],
    dataset: str,
    config: dict[str, object],
    seed: int,
    episode_index: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * episode_index)
    protocol = config["protocol"]
    source_protocol = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))["protocol"]
    normal_train = np.asarray(episode["normal_train"], dtype=float)
    normal_evaluation = np.asarray(episode["normal_evaluation"], dtype=float)
    abnormal_calibration = np.asarray(episode["abnormal_calibration"], dtype=float)
    abnormal_evaluation = np.asarray(episode["abnormal_evaluation"], dtype=float)
    selected, selected_summary = select_features(
        normal_train,
        abnormal_calibration,
        tuple(episode["feature_names"]),
        int(protocol["selected_features"]),
        rng,
    )
    normal_fit = contiguous_fit_block(
        normal_train,
        float(protocol["normal_fit_fraction"]),
        int(protocol["maximum_normal_fit_samples"]),
        rng,
    )[:, selected]
    normal = normal_evaluation[:, selected]
    abnormal = abnormal_evaluation[:, selected]
    abnormal_cal = abnormal_calibration[:, selected]
    sample_period = float(config["dataset_timing"][dataset]["sample_period_seconds"])
    physical_block = float(config["uncertainty"]["physical_block_duration_seconds"])
    block_size = max(2, int(np.ceil(physical_block / sample_period)))
    common = {
        "sample_period_seconds": sample_period,
        "block_size": block_size,
        "draws": int(config["uncertainty"]["draws"]),
        "confidence": float(config["uncertainty"]["confidence"]),
    }

    baseline = MahalanobisAlarm(quantile=0.99).fit(normal_fit)
    m0_normal = baseline.predict(normal)
    m0_abnormal = baseline.predict(abnormal)
    variants: dict[str, object] = {
        "M0": {
            "status": "scored",
            "model": "classical_mahalanobis",
            **score_prediction(
                m0_normal, m0_abnormal, seed=seed + episode_index, **common
            ),
        }
    }

    static = RobustShrinkageMahalanobisAlarm(
        quantile=1.0 - float(protocol["tail_probability"]),
        shrinkage=float(protocol["shrinkage"]),
    ).fit(normal_fit)
    variants["M1"] = {
        "status": "scored",
        "model": "robust_shrinkage_mahalanobis",
        "parameters": {
            "threshold": static.threshold_,
            "covariance_condition": static.covariance_condition_,
        },
        **score_prediction(
            static.predict(normal),
            static.predict(abnormal),
            seed=seed + 100 + episode_index,
            **common,
        ),
    }

    block_model = BlockCalibratedRobustMahalanobisAlarm(
        tail_probability=float(protocol["tail_probability"]),
        shrinkage=float(protocol["shrinkage"]),
        reference_windows=tuple(int(v) for v in protocol["reference_windows"]),
        delays=tuple(int(v) for v in protocol["delays"]),
        validation_fraction=float(protocol["validation_fraction"]),
        block_size=min(block_size, max(2, len(normal_fit) // 4)),
        target_point_false_alarm_rate=float(
            protocol["target_point_false_alarm_rate"]
        ),
        target_block_alarm_rate=float(protocol["target_block_alarm_rate"]),
        block_weight=float(protocol["block_weight"]),
    ).fit(normal_fit)
    variants["M2"] = {
        "status": "scored",
        "model": "block_calibrated_robust_shrinkage_mahalanobis",
        "parameters": {
            "reference_window": block_model.selected_reference_window_,
            "delay": block_model.selected_delay_,
            "threshold": block_model.threshold_,
        },
        **score_prediction(
            block_model.predict(normal),
            block_model.predict(abnormal),
            seed=seed + 200 + episode_index,
            **common,
        ),
    }

    gate_values = config["applicability_thresholds"]
    thresholds = MultivariateApplicabilityThresholds(
        normal_ks_adaptation=float(gate_values["normal_ks_adaptation"]),
        normal_median_shift_adaptation=float(
            gate_values["normal_median_shift_adaptation"]
        ),
        normal_covariance_shift_adaptation=float(
            gate_values["normal_covariance_shift_adaptation"]
        ),
        autocorrelation_block_calibration=float(
            gate_values["autocorrelation_block_calibration"]
        ),
        minimum_block_auc=float(gate_values["minimum_block_auc"]),
        chronological_blocks=int(gate_values["chronological_blocks"]),
    )
    router = AdaptiveMultivariateAlarmRouter(
        applicability_thresholds=thresholds,
        tail_probability=float(protocol["tail_probability"]),
        shrinkage=float(protocol["shrinkage"]),
        reference_windows=tuple(int(v) for v in protocol["reference_windows"]),
        delays=tuple(int(v) for v in protocol["delays"]),
        block_size=min(block_size, max(2, len(normal_fit) // 4)),
    ).fit(normal_fit, abnormal_cal)
    if router.decision_.status == "reject_multivariate":
        variants["M3"] = {
            "status": "denied_multivariate",
            "model": None,
            "decision": router.decision_.as_dict(),
            "empirical": None,
            "event_metrics": None,
            "block_bootstrap": None,
        }
    else:
        variants["M3"] = {
            "status": "scored",
            "model": router.decision_.selected_model,
            "decision": router.decision_.as_dict(),
            **score_prediction(
                router.predict(normal),
                router.predict(abnormal),
                seed=seed + 300 + episode_index,
                **common,
            ),
        }

    source_baseline = MahalanobisAlarm(quantile=0.99).fit(normal_fit)
    source_metrics = binary_alarm_metrics(
        np.r_[np.zeros(len(normal)), np.ones(len(abnormal))],
        np.r_[source_baseline.predict(normal), source_baseline.predict(abnormal)],
    )
    baseline_match = all(
        abs(source_metrics[key] - variants["M0"]["empirical"][key]) < 1e-12
        for key in ("false_alarm_rate", "missed_alarm_rate", "f1")
    )
    return {
        "dataset": dataset,
        "episode_id": episode["id"],
        "selected_features": selected_summary,
        "normal_fit_samples": len(normal_fit),
        "source_chapter3_protocol_match": bool(
            int(source_protocol["selected_features"])
            == int(protocol["selected_features"])
            and baseline_match
        ),
        "split_policy": episode["split_policy"],
        "distribution_audit": {
            "normal_train_to_evaluation": multivariate_distribution_shift(
                normal_train[:, selected], normal
            ).as_dict(),
            "abnormal_calibration_to_evaluation": multivariate_distribution_shift(
                abnormal_cal, abnormal
            ).as_dict(),
            "scope": "held_out_posthoc_diagnostic_not_for_routing",
        },
        "variants": variants,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for variant in ("M0", "M1", "M2", "M3"):
        scored = [row["variants"][variant] for row in rows if row["variants"][variant]["status"] == "scored"]
        denied = len(rows) - len(scored)
        result[variant] = {
            "scored_units": len(scored),
            "denied_units": denied,
            "coverage": len(scored) / len(rows),
            "metrics": {
                name: {
                    "mean": float(np.mean([unit["empirical"][name] for unit in scored])),
                    "standard_deviation": float(
                        np.std([unit["empirical"][name] for unit in scored], ddof=1)
                    )
                    if len(scored) > 1
                    else 0.0,
                }
                for name in ("false_alarm_rate", "missed_alarm_rate", "f1")
            }
            if scored
            else None,
        }
    audits = [row["distribution_audit"]["normal_train_to_evaluation"] for row in rows]
    result["distribution_summary"] = {
        name: {
            "median": float(np.median([audit[name] for audit in audits])),
            "minimum": float(np.min([audit[name] for audit in audits])),
            "maximum": float(np.max([audit[name] for audit in audits])),
        }
        for name in (
            "per_feature_ks_median",
            "per_feature_ks_maximum",
            "standardized_median_shift_maximum",
            "covariance_relative_frobenius_shift",
            "maximum_absolute_correlation_shift",
            "candidate_effective_rank",
            "candidate_absolute_lag_one_median",
        )
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    run_name = out_dir.name
    if run_name not in config["runs"]:
        raise ValueError(f"out_dir basename must be one of {sorted(config['runs'])}")
    seed = int(config["runs"][run_name]["seed"])
    print(
        "This run tests whether robust scaling, covariance shrinkage, and "
        "chronological block calibration improve held-out multivariate transfer."
    )
    started = datetime.now(timezone.utc)
    before = time.perf_counter()
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    datasets, paths = load_episodes(source_config)
    episode_results = {
        dataset: [
            evaluate_episode(episode, dataset, config, seed, index)
            for index, episode in enumerate(episodes)
        ]
        for dataset, episodes in datasets.items()
    }
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
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - before,
        "result": {
            "seed": seed,
            "datasets": episode_results,
            "aggregate_metrics": {
                dataset: aggregate(rows) for dataset, rows in episode_results.items()
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
                "aggregate_metrics": payload["result"]["aggregate_metrics"],
                "wall_clock_seconds": payload["wall_clock_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
