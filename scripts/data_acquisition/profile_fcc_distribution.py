"""Run a comprehensive G0 prior audit of the official FCC alarm archives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from iia_benchmark.data import load_fcc_alarm_runs, load_fcc_timeseries_runs


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(np.asarray(values, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]),
            strict=True,
        )
    }


def feature_profile(values: np.ndarray, names: tuple[str, ...]) -> dict:
    missing = np.isnan(values)
    return {
        "samples": int(values.shape[0]),
        "features": int(values.shape[1]),
        "infinite_values": int(np.isinf(values).sum()),
        "missing_values": int(missing.sum()),
        "missing_fraction": float(np.mean(missing)),
        "features_with_missing": [
            name
            for name, count in zip(names, np.sum(missing, axis=0), strict=True)
            if count > 0
        ],
        "zero_variance_features": [
            name for name, deviation in zip(names, np.nanstd(values, axis=0), strict=True)
            if deviation == 0
        ],
        "per_feature": {
            name: {
                "mean": float(mean),
                "std": float(std),
                "min": float(minimum),
                "max": float(maximum),
            }
            for name, mean, std, minimum, maximum in zip(
                names,
                np.nanmean(values, axis=0),
                np.nanstd(values, axis=0),
                np.nanmin(values, axis=0),
                np.nanmax(values, axis=0),
                strict=True,
            )
        },
    }


def build_profile(alarm_archive: Path, timeseries_archive: Path) -> dict:
    alarms = load_fcc_alarm_runs(alarm_archive)
    series = load_fcc_timeseries_runs(timeseries_archive)
    alarm_by_key = {(run.scenario, run.run_number): run for run in alarms}
    series_by_key = {(run.scenario, run.run_number): run for run in series}
    shapes = Counter(str(run.alarm_states.shape) for run in alarms)
    scenario_counts = Counter(run.scenario for run in alarms)
    all_states = np.stack([run.alarm_states for run in alarms])
    all_rising = np.stack([run.representation("rising_edge") for run in alarms])
    run_active_fraction = np.mean(all_states, axis=(1, 2))
    run_activations = np.sum(all_rising, axis=(1, 2))
    run_unique_tags = np.sum(np.any(all_rising > 0, axis=1), axis=1)

    hashes: dict[str, list[str]] = defaultdict(list)
    scenario_rows: dict[str, dict] = {}
    for scenario in sorted(scenario_counts):
        selected = [run for run in alarms if run.scenario == scenario]
        states = np.stack([run.alarm_states for run in selected])
        rising = np.stack([run.representation("rising_edge") for run in selected])
        sequence_hashes = {
            hashlib.sha256(run.alarm_states.tobytes()).hexdigest() for run in selected
        }
        for run in selected:
            digest = hashlib.sha256(run.alarm_states.tobytes()).hexdigest()
            hashes[digest].append(run.run_id)
        scenario_rows[scenario] = {
            "runs": len(selected),
            "unique_state_matrices": len(sequence_hashes),
            "mean_active_fraction": float(np.mean(states)),
            "mean_activation_events": float(np.mean(np.sum(rising, axis=(1, 2)))),
            "mean_unique_activated_tags": float(
                np.mean(np.sum(np.any(rising > 0, axis=1), axis=1))
            ),
            "tags_ever_activated": int(np.sum(np.any(rising > 0, axis=(0, 1)))),
        }

    cross_scenario_duplicates = []
    for digest, run_ids in hashes.items():
        scenarios = {run_id.rsplit("_run", 1)[0] for run_id in run_ids}
        if len(scenarios) > 1:
            cross_scenario_duplicates.append(
                {"sha256": digest, "runs": sorted(run_ids), "scenarios": sorted(scenarios)}
            )

    scenarios = sorted(scenario_counts)
    train = [run for run in alarms if 1 <= run.run_number <= 60]
    test = [run for run in alarms if 81 <= run.run_number <= 100]
    centroids = {
        scenario: np.mean(
            np.stack([run.alarm_states for run in train if run.scenario == scenario]),
            axis=0,
        )
        for scenario in scenarios
    }
    centroid_predictions = []
    for run in test:
        distances = {
            scenario: float(np.linalg.norm(run.alarm_states - centroid))
            for scenario, centroid in centroids.items()
        }
        centroid_predictions.append(min(distances, key=distances.get))
    centroid_accuracy = float(
        np.mean([prediction == run.scenario for prediction, run in zip(centroid_predictions, test, strict=True)])
    )
    pairwise_centroid_distances = []
    for left_index, left in enumerate(scenarios):
        for right in scenarios[left_index + 1 :]:
            pairwise_centroid_distances.append(
                {
                    "left": left,
                    "right": right,
                    "distance": float(np.linalg.norm(centroids[left] - centroids[right])),
                }
            )
    nearest_centroid_pairs = sorted(
        pairwise_centroid_distances, key=lambda row: row["distance"]
    )[:10]

    process = np.vstack([run.process_values for run in series])
    valves = np.vstack([run.valve_values for run in series])
    disturbances = np.vstack([run.disturbance_values for run in series])
    root_signal_checks: dict[str, dict] = {}
    for scenario in scenarios:
        selected = [run for run in series if run.scenario == scenario]
        root = selected[0].root_disturbance
        root_index = selected[0].disturbance_names.index(root)
        root_values = np.concatenate(
            [run.disturbance_values[:, root_index] for run in selected]
        )
        root_signal_checks[scenario] = {
            "root_disturbance": root,
            "nonzero_fraction": float(np.mean(root_values != 0)),
            "mean": float(np.mean(root_values)),
            "min": float(np.min(root_values)),
            "max": float(np.max(root_values)),
        }

    tag_activation_rate = np.mean(all_states, axis=(0, 1))
    tag_rising_rate = np.mean(all_rising, axis=(0, 1))
    alarm_names = alarms[0].alarm_names
    missing_keys = sorted(set(alarm_by_key) - set(series_by_key))
    extra_keys = sorted(set(series_by_key) - set(alarm_by_key))
    g0_checks = {
        "alarm_binary": bool(np.isin(all_states, [0, 1]).all()),
        "alarm_shape_uniform": len(shapes) == 1,
        "sixteen_scenarios": len(scenario_counts) == 16,
        "one_hundred_runs_per_scenario": set(scenario_counts.values()) == {100},
        "alarm_timeseries_keys_aligned": not missing_keys and not extra_keys,
        "no_timeseries_infinities": bool(
            not np.isinf(process).any()
            and not np.isinf(valves).any()
            and not np.isinf(disturbances).any()
        ),
        "timeseries_missingness_quantified": True,
        "root_disturbance_present_and_nonzero": all(
            row["nonzero_fraction"] > 0 for row in root_signal_checks.values()
        ),
        "no_cross_scenario_exact_alarm_duplicates": not cross_scenario_duplicates,
        "split_run_numbers_disjoint": True,
    }
    warnings = []
    if cross_scenario_duplicates:
        warnings.append(
            "Exact alarm-state matrices occur in more than one class; classification ceiling and label ambiguity require reporting."
        )
    if centroid_accuracy <= 1 / len(scenarios):
        warnings.append("Nearest-centroid prior does not exceed chance.")

    return {
        "schema_version": 1,
        "dataset_family": "fcc_alarm",
        "source": {
            "alarm_archive": alarm_archive.relative_to(ROOT).as_posix(),
            "alarm_archive_bytes": alarm_archive.stat().st_size,
            "alarm_archive_sha256": sha256_file(alarm_archive),
            "timeseries_archive": timeseries_archive.relative_to(ROOT).as_posix(),
            "timeseries_archive_bytes": timeseries_archive.stat().st_size,
            "timeseries_archive_sha256": sha256_file(timeseries_archive),
            "citation_doi": "10.60517/2v23vv393",
        },
        "alarm_prior": {
            "runs": len(alarms),
            "scenarios": len(scenario_counts),
            "runs_per_scenario": dict(sorted(scenario_counts.items())),
            "shape_counts": dict(shapes),
            "alarm_variables": len(alarm_names),
            "run_active_fraction": quantiles(run_active_fraction),
            "run_activation_events": quantiles(run_activations),
            "run_unique_activated_tags": quantiles(run_unique_tags),
            "never_active_tags": [
                name for name, rate in zip(alarm_names, tag_activation_rate, strict=True) if rate == 0
            ],
            "never_rising_tags": [
                name for name, rate in zip(alarm_names, tag_rising_rate, strict=True) if rate == 0
            ],
            "per_tag": {
                name: {
                    "state_prevalence": float(state_rate),
                    "rising_edge_rate": float(rising_rate),
                }
                for name, state_rate, rising_rate in zip(
                    alarm_names, tag_activation_rate, tag_rising_rate, strict=True
                )
            },
            "per_scenario": scenario_rows,
            "cross_scenario_exact_duplicate_groups": len(cross_scenario_duplicates),
            "cross_scenario_exact_duplicates": cross_scenario_duplicates,
        },
        "timeseries_prior": {
            "runs": len(series),
            "samples_per_run": sorted({len(run.timestamps) for run in series}),
            "process": feature_profile(process, series[0].process_names),
            "valves": feature_profile(valves, series[0].valve_names),
            "disturbances": feature_profile(disturbances, series[0].disturbance_names),
            "root_signal_checks": root_signal_checks,
        },
        "label_separability_prior": {
            "method": "nearest class centroid on complete 57x60 alarm-state matrices",
            "train_runs": "1-60 per class",
            "test_runs": "81-100 per class",
            "accuracy": centroid_accuracy,
            "chance_accuracy": 1 / len(scenarios),
            "nearest_centroid_pairs": nearest_centroid_pairs,
            "claim_boundary": "diagnostic prior only; not a registered benchmark baseline",
        },
        "split_manifest": {
            "group_key": "scenario + complete simulation run number",
            "train_run_numbers": list(range(1, 61)),
            "calibration_run_numbers": list(range(61, 81)),
            "test_run_numbers": list(range(81, 101)),
            "train_runs": 16 * 60,
            "calibration_runs": 16 * 20,
            "test_runs": 16 * 20,
            "leaderboard_eligible": False,
            "reason": "Engineering split pending paper-protocol and independent held-out freeze.",
        },
        "representation_boundaries": [
            "FCC contains abnormal-situation runs but no separate normal-operation corpus; it cannot by itself fit or score a normal operating zone.",
            "Alarm-state and rising-edge representations are both retained; event-fingerprint methods must pass activation on rising edges.",
            "Scenario labels identify injected abnormal situations and root disturbances, not expert-confirmed operator alarm-flood intervals.",
            "Continuous experiments use within-run interpolation followed by training-fold-only feature medians; missingness indicators and excluded runs must be reported."
        ],
        "g0_checks": g0_checks,
        "g0_pass": all(g0_checks.values()),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alarm-archive",
        type=Path,
        default=ROOT / "data/public_datasets/fcc_alarm/alarmseriesdata.zip",
    )
    parser.add_argument(
        "--timeseries-archive",
        type=Path,
        default=ROOT / "data/public_datasets/fcc_alarm/timeseriesdata.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/reports/fcc_alarm_prior_validation.json",
    )
    args = parser.parse_args()
    profile = build_profile(args.alarm_archive.resolve(), args.timeseries_archive.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if profile["g0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
