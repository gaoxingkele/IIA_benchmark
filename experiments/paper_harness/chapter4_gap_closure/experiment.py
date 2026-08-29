#!/usr/bin/env python3
"""Close the remaining Chapter 4 activation/evidence gaps without hiding negatives."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import (  # noqa: E402
    load_enas_event_log,
    load_imaks_causal_edges,
    load_imaks_sensor_data,
)
from iia_benchmark.models import (  # noqa: E402
    PLRContributionRCA,
    RecursiveBayesianAlarmRCA,
    cluster_information_granules,
    clustered_surrogate_threshold,
    discrete_direct_transfer_entropy,
    discrete_transfer_entropy,
    information_granulation_direct_transfer_entropy,
    information_granulation_transfer_entropy,
    information_granules,
)


CONFIG = ROOT / "configs/experiments/chapter4_gap_closure.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(data_paths: Sequence[Path]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sources = [
        ROOT / "src/iia_benchmark/models/root_cause_book.py",
        ROOT / "src/iia_benchmark/data/enas.py",
        ROOT / "src/iia_benchmark/data/imaks.py",
        Path(__file__),
        CONFIG,
        *data_paths,
    ]
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources
        },
    }


def persist_impulses(values: np.ndarray, rows: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.int8)
    for index in np.flatnonzero(values):
        output[index : min(index + rows, len(output))] = 1
    return output


def evaluate_recursive_bn(
    cause_states: np.ndarray,
    alarm_states: np.ndarray,
    cause_names: Sequence[str],
    event_indices: np.ndarray,
    *,
    response_time: int,
    persistence: int,
) -> dict[str, Any]:
    model = RecursiveBayesianAlarmRCA(
        tuple(cause_names),
        response_time_samples=response_time,
        initial_probability=0.0,
    )
    decisions = model.infer_sequence(cause_states, alarm_states)
    evaluation_indices = np.minimum(event_indices + persistence - 1, len(alarm_states) - 1)
    event_decisions = [decisions[int(index)] for index in evaluation_indices]
    posterior = model.posterior_patterns(1)
    return {
        "events": int(len(event_indices)),
        "nonempty_decision_rate": float(np.mean([bool(value) for value in event_decisions])),
        "unknown_decision_rate": float(
            np.mean([value == ("unknown",) for value in event_decisions])
        ),
        "known_candidate_decision_rate": float(
            np.mean(
                [
                    bool(set(value) & set(cause_names)) and value != ("unknown",)
                    for value in event_decisions
                ]
            )
        ),
        "unique_event_decisions": len(set(event_decisions)),
        "posterior_entropy": float(
            -np.sum(posterior[posterior > 0] * np.log(posterior[posterior > 0]))
        ),
        "posterior_range": float(np.max(posterior) - np.min(posterior)),
        "top_decisions": [
            {"decision": list(decision), "count": count}
            for decision, count in Counter(event_decisions).most_common(8)
        ],
    }


def run_enas(specification: dict[str, Any], seed: int) -> dict[str, Any]:
    path = ROOT / specification["path"]
    data = load_enas_event_log(path)
    persistence = int(specification["event_persistence_rows"])
    response_time = int(specification["response_time_samples"])
    rng = np.random.default_rng(seed)
    targets: dict[str, Any] = {}
    for target, cause_names in specification["targets"].items():
        cause_states = np.column_stack([data.signal(name) for name in cause_names])
        raw_alarm = data.error(target)
        persistent_alarm = persist_impulses(raw_alarm, persistence)
        event_indices = np.flatnonzero(raw_alarm)
        raw = evaluate_recursive_bn(
            cause_states,
            raw_alarm,
            cause_names,
            event_indices,
            response_time=response_time,
            persistence=1,
        )
        persistent = evaluate_recursive_bn(
            cause_states,
            persistent_alarm,
            cause_names,
            event_indices,
            response_time=response_time,
            persistence=persistence,
        )
        corrupt = cause_states.copy()
        flips = rng.random(corrupt.shape) < float(specification["corruption_probability"])
        corrupt[flips] = 1 - corrupt[flips]
        corrupted = evaluate_recursive_bn(
            corrupt,
            persistent_alarm,
            cause_names,
            event_indices,
            response_time=response_time,
            persistence=persistence,
        )
        targets[target] = {
            "cause_names": cause_names,
            "manual_annotation_rows": int(np.sum(raw_alarm)),
            "raw_impulse": raw,
            "persistent_adapter": persistent,
            "corrupted_persistent_adapter": corrupted,
            "input_bit_flips": int(np.sum(flips)),
        }
    checks = {
        "finite_strict_timestamps": bool(
            np.isfinite(data.timestamps).all() and np.all(np.diff(data.timestamps) > 0)
        ),
        "binary_signals_and_errors": bool(
            np.isin(data.signal_states, (0, 1)).all()
            and np.isin(data.error_states, (0, 1)).all()
        ),
        "all_error_types_observed": all(np.sum(data.error(name)) > 0 for name in data.error_names),
        "candidate_columns_present": all(
            name in data.signal_names
            for names in specification["targets"].values()
            for name in names
        ),
        "manual_markers_are_impulses": all(
            not np.any(data.error(name)[1:] & data.error(name)[:-1])
            for name in data.error_names
        ),
    }
    mechanism = all(
        row["persistent_adapter"]["nonempty_decision_rate"] > 0
        and row["persistent_adapter"]["posterior_range"] > 0
        for row in targets.values()
    )
    return {
        "dataset_family": "enas",
        "match": specification["match"],
        "protocol": specification["protocol"],
        "prior_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "rows": len(data.timestamps),
            "signals": len(data.signal_names),
            "error_counts": {
                name: int(np.sum(data.error(name))) for name in data.error_names
            },
            "timestamp_span_days": float(
                (data.timestamps[-1] - data.timestamps[0]) / 86_400.0
            ),
        },
        "targets": targets,
        "gates": {
            "mechanism": mechanism,
            "performance": None,
            "competitive": None,
            "decision": "E2 engineering activation only; tag-level truth is absent",
        },
        "boundary": specification["boundary"],
    }


def select_transition_windows(
    frame: pd.DataFrame,
    *,
    fold: int,
    seed: int,
    half_window: int,
    count: int,
) -> list[tuple[int, int, int]]:
    labels = frame["type"].astype(str).to_numpy()
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    block_start = len(frame) * fold // 3
    block_stop = len(frame) * (fold + 1) // 3
    candidates = transitions[
        (transitions >= block_start + half_window)
        & (transitions <= block_stop - half_window)
    ]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for center in candidates[rng.permutation(len(candidates))]:
        if all(abs(int(center) - other) >= 2 * half_window for other in selected):
            selected.append(int(center))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError("PIADE fold lacks enough non-overlapping transition windows")
    return sorted((center - half_window, center, center + half_window) for center in selected)


def run_piade(specification: dict[str, Any], seed: int, fold: int) -> dict[str, Any]:
    path = ROOT / specification["path"]
    columns = [
        "equipment_ID",
        "start",
        "type",
        *specification["predictors"],
        specification["target"],
    ]
    frame = pd.read_csv(path, usecols=columns)
    half = int(specification["half_window_rows"])
    equipment_results = {}
    all_rows = []
    input_ranges = []
    for equipment in specification["equipment_ids"]:
        subset = (
            frame.loc[frame["equipment_ID"] == equipment]
            .sort_values("start", kind="stable")
            .dropna(subset=[*specification["predictors"], specification["target"]])
            .reset_index(drop=True)
        )
        windows = select_transition_windows(
            subset,
            fold=fold,
            seed=seed,
            half_window=half,
            count=int(specification["windows_per_equipment"]),
        )
        rows = []
        for start, center, stop in windows:
            values = subset.iloc[start:stop]
            result = PLRContributionRCA(
                max_segments=2,
                min_size=max(8, half // 2),
                max_lag=int(specification["max_lag"]),
            ).analyze(
                values.loc[:, specification["predictors"]].to_numpy(dtype=float),
                values[specification["target"]].to_numpy(dtype=float),
                segment_boundaries=(0, half, 2 * half),
            )
            factors = np.asarray([segment.factors for segment in result], dtype=float)
            active = bool(np.any(np.sum(factors, axis=1) > 0))
            row = {
                "source_start": float(values["start"].iloc[0]),
                "source_stop": float(values["start"].iloc[-1]),
                "transition_from": str(subset["type"].iloc[center - 1]),
                "transition_to": str(subset["type"].iloc[center]),
                "lags": list(result[0].lags),
                "target_trends": [segment.target_trend for segment in result],
                "source_trends": [list(segment.source_trends) for segment in result],
                "factors": factors.tolist(),
                "active": active,
                "finite_nonnegative": bool(
                    np.isfinite(factors).all() and np.all(factors >= 0)
                ),
            }
            rows.append(row)
            all_rows.append(row)
            input_ranges.append(
                {
                    "equipment": equipment,
                    "start": row["source_start"],
                    "stop": row["source_stop"],
                }
            )
        active_rows = [row for row in rows if row["active"]]
        factor_rows = np.asarray(
            [factor for row in active_rows for factor in row["factors"] if sum(factor) > 0],
            dtype=float,
        )
        equipment_results[equipment] = {
            "source_rows": len(subset),
            "windows": len(rows),
            "active_windows": len(active_rows),
            "activation_rate": float(len(active_rows) / len(rows)),
            "mean_active_factors": (
                np.mean(factor_rows, axis=0).tolist() if len(factor_rows) else None
            ),
            "unique_transition_pairs": len(
                {(row["transition_from"], row["transition_to"]) for row in rows}
            ),
        }
    activation_rate = float(np.mean([row["active"] for row in all_rows]))
    checks = {
        "five_equipment_groups": set(equipment_results) == set(specification["equipment_ids"]),
        "finite_nonnegative_factors": all(row["finite_nonnegative"] for row in all_rows),
        "nonoverlapping_within_run": all(
            left["equipment"] != right["equipment"]
            or left["stop"] < right["start"]
            for index, left in enumerate(input_ranges)
            for right in input_ranges[index + 1 :]
        ),
        "state_transitions_present": all(
            row["transition_from"] != row["transition_to"] for row in all_rows
        ),
    }
    return {
        "dataset_family": "piade",
        "match": specification["match"],
        "protocol": specification["protocol"],
        "chronological_fold": fold,
        "prior_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "evaluated_windows": len(all_rows),
            "input_ranges": input_ranges,
        },
        "equipment": equipment_results,
        "metrics": {
            "activation_rate": activation_rate,
            "active_windows": int(sum(row["active"] for row in all_rows)),
            "evaluated_windows": len(all_rows),
            "unique_lag_vectors": len({tuple(row["lags"]) for row in all_rows}),
        },
        "gates": {
            "mechanism": activation_rate >= float(specification["minimum_active_rate"]),
            "performance": None,
            "competitive": None,
            "decision": "E2 real-data contribution activation; causal ranking score denied",
        },
        "boundary": specification["boundary"],
    }


def run_controlled_igdte(specification: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    length = int(specification["length"])
    lag = int(specification["path_lag"])
    root = rng.binomial(1, float(specification["event_probability"]), length)
    middle = np.zeros(length, dtype=int)
    target = np.zeros(length, dtype=int)
    middle[lag:] = root[:-lag]
    target[lag:] = middle[:-lag]
    window = int(specification["granule_window_samples"])
    noise = float(specification["continuous_noise_standard_deviation"])
    continuous = [
        np.repeat(values, window) + rng.normal(0.0, noise, length * window)
        for values in (root, middle, target)
    ]
    labels = [
        cluster_information_granules(information_granules(values, window), min_samples=4)
        for values in continuous
    ]
    source_lag = 2 * lag
    threshold = clustered_surrogate_threshold(
        labels[0],
        labels[2],
        lag=source_lag,
        order=2,
        simulations=int(specification["surrogate_simulations"]),
        significance=float(specification["significance"]),
        seed=seed,
    )
    igte = information_granulation_transfer_entropy(
        continuous[0],
        continuous[2],
        window_size=window,
        lag=source_lag,
        order=2,
        min_samples=4,
    )
    igdte = information_granulation_direct_transfer_entropy(
        continuous[0],
        continuous[2],
        continuous[1],
        window_size=window,
        source_lag=source_lag,
        intermediate_lag=lag,
        order=2,
        min_samples=4,
    )
    return {
        "match": specification["match"],
        "protocol": specification["protocol"],
        "metrics": {
            "igte": igte,
            "igdte": igdte,
            "surrogate_threshold": threshold,
            "conditional_to_pairwise_ratio": float(igdte / igte) if igte else None,
            "source_lag": source_lag,
            "intermediate_lag": lag,
            "cluster_counts": [len(set(values)) for values in labels],
        },
        "gates": {
            "pairwise_edge_active": igte > threshold,
            "indirect_edge_pruned": igdte <= threshold,
            "mechanism": igte > threshold and igdte <= threshold,
            "performance": None,
            "competitive": None,
            "decision": "E1 mechanism activation only",
        },
        "boundary": specification["boundary"],
    }


def run_imaks(specification: dict[str, Any], seed: int) -> dict[str, Any]:
    path = ROOT / specification["path"]
    data = load_imaks_sensor_data(path)
    edges = load_imaks_causal_edges(path)
    start = pd.Timestamp(specification["window_start"]).timestamp()
    stop = pd.Timestamp(specification["window_stop"]).timestamp()
    mask = (data.timestamps >= start) & (data.timestamps <= stop)
    source = data.series(specification["source_sensor"])[mask]
    target = data.series(specification["target_sensor"])[mask]
    control = data.series(specification["control_sensor"])[mask]
    source_state = data.anomaly_state(specification["source_sensor"])[mask]
    target_state = data.anomaly_state(specification["target_sensor"])[mask]
    model = RecursiveBayesianAlarmRCA(
        [specification["source_sensor"]], response_time_samples=5, initial_probability=0.0
    )
    decisions = model.infer_sequence(source_state[:, None], target_state)
    target_indices = np.flatnonzero(target_state)
    bn_recall = float(
        np.mean(
            [
                decisions[index] == (specification["source_sensor"],)
                for index in target_indices
            ]
        )
    )
    granule_window = int(specification["granule_window_samples"])
    granular_lag = int(
        round(
            float(specification["source_target_lag_minutes"])
            * 60.0
            / data.sample_seconds
            / granule_window
        )
    )
    labels = [
        cluster_information_granules(
            information_granules(values, granule_window), min_samples=4
        )
        for values in (source, target, control)
    ]
    threshold = clustered_surrogate_threshold(
        labels[0],
        labels[1],
        lag=granular_lag,
        order=2,
        simulations=int(specification["surrogate_simulations"]),
        significance=float(specification["significance"]),
        seed=seed,
    )
    igte = discrete_transfer_entropy(
        labels[0], labels[1], lag=granular_lag, source_horizon=2, target_horizon=2
    )
    igdte = discrete_direct_transfer_entropy(
        labels[0],
        labels[1],
        labels[2],
        source_lag=granular_lag,
        intermediate_lag=1,
        source_horizon=2,
        target_horizon=2,
        intermediate_horizon=2,
    )
    plr = PLRContributionRCA(max_segments=3, min_size=30, max_lag=200).analyze(
        np.column_stack([source, control]),
        target,
        segment_boundaries=(0, 240, 361, len(target)),
    )
    plr_factors = np.asarray([row.factors for row in plr])
    named_edge = any(
        edge.source == specification["source_sensor"]
        and edge.target == specification["target_sensor"]
        and edge.relation == "correlates_with"
        for edge in edges
    )
    checks = {
        "regular_finite_grid": bool(
            np.isfinite(data.values).all()
            and np.allclose(np.diff(data.timestamps), data.sample_seconds)
        ),
        "registered_causal_edge_present": named_edge,
        "source_and_target_events_present": bool(np.any(source_state) and np.any(target_state)),
        "declared_lag_matches_onsets": int(np.flatnonzero(target_state)[0] - np.flatnonzero(source_state)[0])
        == int(round(float(specification["source_target_lag_minutes"]) * 60 / data.sample_seconds)),
    }
    return {
        "dataset_family": "imaks",
        "match": specification["match"],
        "protocol": specification["protocol"],
        "prior_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "timestamps": len(data.timestamps),
            "sensors": len(data.sensor_names),
            "window_samples": int(np.sum(mask)),
        },
        "recursive_bn": {
            "target_alarm_samples": int(len(target_indices)),
            "source_cause_recall_during_target_alarm": bn_recall,
            "decision_at_target_onset": list(decisions[int(target_indices[0])]),
            "decision_at_target_end": list(decisions[int(target_indices[-1])]),
        },
        "igdte": {
            "known_edge_igte": float(igte),
            "control_conditioned_igdte": float(igdte),
            "surrogate_threshold": float(threshold),
            "known_edge_detected": bool(igte > threshold),
            "granular_lag": granular_lag,
            "cluster_counts": [len(set(values)) for values in labels],
        },
        "plr": {
            "target_trends": [row.target_trend for row in plr],
            "lags": list(plr[0].lags),
            "factors": plr_factors.tolist(),
            "active": bool(np.any(np.sum(plr_factors, axis=1) > 0)),
            "interpretation": "The documented anomalous interval is a sustained offset: its target trend is zero, while only the post-event recovery segment activates. This does not recover the stated 90-minute causal delay.",
        },
        "gates": {
            "mechanism": True,
            "performance": None,
            "competitive": None,
            "decision": "synthetic transfer diagnostics only",
        },
        "boundary": specification["boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if out_dir.name not in config["runs"]:
        raise ValueError(f"out_dir basename must be one of {sorted(config['runs'])}")
    run = config["runs"][out_dir.name]
    seed = int(run["seed"])
    data_paths = [
        ROOT / config["datasets"][name]["path"] for name in ("enas", "piade", "imaks")
    ]
    started = datetime.now(timezone.utc)
    execution_provenance = provenance(data_paths)
    before = time.perf_counter()
    results = {
        "controlled_igdte": run_controlled_igdte(config["datasets"]["controlled_chain"], seed),
        "enas_recursive_bn": run_enas(config["datasets"]["enas"], seed),
        "piade_plr": run_piade(
            config["datasets"]["piade"], seed, int(run["chronological_fold"])
        ),
        "imaks_diagnostics": run_imaks(config["datasets"]["imaks"], seed),
    }
    mandatory = {
        "controlled_igdte_mechanism": results["controlled_igdte"]["gates"]["mechanism"],
        "enas_prior": results["enas_recursive_bn"]["prior_gate"]["passed"],
        "enas_mechanism": results["enas_recursive_bn"]["gates"]["mechanism"],
        "piade_prior": results["piade_plr"]["prior_gate"]["passed"],
        "piade_mechanism": results["piade_plr"]["gates"]["mechanism"],
        "imaks_prior": results["imaks_diagnostics"]["prior_gate"]["passed"],
    }
    if not all(mandatory.values()):
        raise RuntimeError(f"mandatory gate failed; final_info withheld: {mandatory}")
    duration = time.perf_counter() - before
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "experiment_id": config["id"],
        "run_name": out_dir.name,
        "seed": seed,
        "chronological_fold": int(run["chronological_fold"]),
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG),
        "git_revision": revision,
        "execution_provenance": execution_provenance,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": duration,
        "mandatory_gates": mandatory,
        "results": results,
        "reporting_boundary": (
            "IGDTE receives controlled activation only; recursive BN receives real EnAS engineering activation; "
            "PLR receives real grouped PIADE contribution activation. No unavailable paper or plant score is claimed."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run": out_dir.name,
                "seed": seed,
                "wall_clock_seconds": duration,
                "mandatory_gates": mandatory,
                "piade_plr_activation_rate": results["piade_plr"]["metrics"]["activation_rate"],
                "imaks_known_edge_detected": results["imaks_diagnostics"]["igdte"]["known_edge_detected"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
