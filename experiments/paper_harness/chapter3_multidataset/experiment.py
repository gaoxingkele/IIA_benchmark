#!/usr/bin/env python3
"""Run Book Chapter 3 named-item and multi-dataset validation entries."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from scipy.stats import ks_2samp, kstest


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
    AdaptiveTimeGradient,
    BayesianWindowRegressionAlarm,
    CondenserNOZAlarm,
    CondenserParameters,
    CondenserPhysicalModel,
    ConvexHullNOZAlarm,
    MahalanobisAlarm,
    SearchConeNOZAlarm,
    VariationDirectionAlarm,
    condenser_alarm_rate_bounds,
    convex_hull_fitness_index,
)


CONFIG = ROOT / "configs/experiments/book_chapter3_multidataset.json"


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
        "src/iia_benchmark/models/multivariate.py",
        "src/iia_benchmark/models/multivariate_book.py",
        "src/iia_benchmark/data/tep.py",
        "src/iia_benchmark/data/pronto.py",
        "src/iia_benchmark/data/skab.py",
        "experiments/paper_harness/chapter3_multidataset/experiment.py",
        "configs/experiments/book_chapter3_multidataset.json",
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


def distribution_prior(
    normal_train: np.ndarray,
    normal_evaluation: np.ndarray,
    abnormal_calibration: np.ndarray,
    abnormal_evaluation: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, object]:
    blocks = [normal_train, normal_evaluation, abnormal_calibration, abnormal_evaluation]
    constant = np.std(normal_train, axis=0, ddof=1) <= 1e-12
    ks_values = np.asarray(
        [
            ks_2samp(normal_train[:, index], normal_evaluation[:, index]).statistic
            for index in range(normal_train.shape[1])
            if not constant[index]
        ],
        dtype=float,
    )
    scale = np.std(normal_train, axis=0, ddof=1)
    valid = scale > 1e-12
    normal_shift = np.zeros(normal_train.shape[1], dtype=float)
    normal_shift[valid] = np.abs(
        np.median(normal_evaluation[:, valid], axis=0)
        - np.median(normal_train[:, valid], axis=0)
    ) / scale[valid]
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
            name for index, name in enumerate(feature_names) if constant[index]
        ],
        "normal_train_evaluation_ks_median": float(np.median(ks_values)),
        "normal_train_evaluation_ks_maximum": float(np.max(ks_values)),
        "normal_standardized_median_shift_maximum": float(np.max(normal_shift)),
        "partition_overlap": 0,
    }


def load_episodes(config: dict[str, object]) -> tuple[dict[str, list[dict[str, object]]], list[Path]]:
    fraction = float(config["protocol"]["abnormal_calibration_fraction"])
    datasets: dict[str, list[dict[str, object]]] = {}
    paths: list[Path] = []

    tep = config["datasets"]["tep_classic"]
    normal_train_path = ROOT / tep["normal_train"]
    normal_evaluation_path = ROOT / tep["normal_evaluation"]
    normal_train = load_tep_ascii(normal_train_path)
    normal_evaluation = load_tep_ascii(normal_evaluation_path)
    episodes = []
    for relative in tep["fault_runs"]:
        path = ROOT / relative
        run = load_tep_ascii(path, fault_start=int(tep["fault_start"]))
        abnormal_calibration, abnormal_evaluation = split_abnormal(
            run.values[int(tep["fault_start"]) :], fraction
        )
        episodes.append(
            {
                "id": path.stem,
                "feature_names": run.feature_names,
                "normal_train": normal_train.values,
                "normal_evaluation": normal_evaluation.values,
                "abnormal_calibration": abnormal_calibration,
                "abnormal_evaluation": abnormal_evaluation,
                "split_policy": "separate normal train/evaluation files; fault run split chronologically after sample 160",
                "prior": distribution_prior(
                    normal_train.values,
                    normal_evaluation.values,
                    abnormal_calibration,
                    abnormal_evaluation,
                    run.feature_names,
                ),
            }
        )
        paths.append(path)
    paths.extend([normal_train_path, normal_evaluation_path])
    datasets["tep_classic"] = episodes

    pronto = config["datasets"]["pronto"]
    pronto_paths = [ROOT / relative for relative in pronto["paths"]]
    pronto_runs = [load_pronto_merged_csv(path) for path in pronto_paths]
    episodes = []
    for selected in pronto["selected_fault_segments"]:
        run = pronto_runs[int(selected["path_index"])]
        normal_train_mask, normal_evaluation_mask = pronto_normal_train_evaluation_masks(
            run.labels,
            train_fraction=float(config["protocol"]["normal_train_fraction"]),
            purge_samples=int(config["protocol"]["normal_purge_samples"]),
        )
        start, stop = segment_bounds(
            run.labels, str(selected["label"]), int(selected["occurrence"])
        )
        abnormal_calibration, abnormal_evaluation = split_abnormal(
            run.process_values[start:stop], fraction
        )
        nt = run.process_values[normal_train_mask]
        ne = run.process_values[normal_evaluation_mask & (run.labels == "Normal")]
        episodes.append(
            {
                "id": f"{run.run_id}_{selected['label'].replace(' ', '_')}_{selected['occurrence']}",
                "feature_names": run.process_names,
                "normal_train": nt,
                "normal_evaluation": ne,
                "abnormal_calibration": abnormal_calibration,
                "abnormal_evaluation": abnormal_evaluation,
                "split_policy": "purged chronological normal split plus chronological 40/60 fault-segment split",
                "prior": distribution_prior(
                    nt, ne, abnormal_calibration, abnormal_evaluation, run.process_names
                ),
            }
        )
    paths.extend(pronto_paths)
    datasets["pronto"] = episodes

    skab = config["datasets"]["skab"]
    skab_normal_path = ROOT / skab["normal_train"]
    skab_normal = load_skab_csv(skab_normal_path)
    boundary = int(
        np.floor(len(skab_normal.values) * float(config["protocol"]["normal_train_fraction"]))
    )
    episodes = []
    for relative in skab["fault_runs"]:
        path = ROOT / relative
        run = load_skab_csv(path)
        abnormal_calibration, abnormal_evaluation = split_abnormal(
            run.values[run.abnormal], fraction
        )
        nt = skab_normal.values[:boundary]
        ne = skab_normal.values[boundary:]
        episodes.append(
            {
                "id": f"{path.parent.name}_{path.stem}",
                "feature_names": run.feature_names,
                "normal_train": nt,
                "normal_evaluation": ne,
                "abnormal_calibration": abnormal_calibration,
                "abnormal_evaluation": abnormal_evaluation,
                "split_policy": "anomaly-free file chronological 50/50 split; anomaly samples chronological 40/60 split",
                "prior": distribution_prior(
                    nt, ne, abnormal_calibration, abnormal_evaluation, run.feature_names
                ),
            }
        )
        paths.append(path)
    paths.append(skab_normal_path)
    datasets["skab"] = episodes
    return datasets, sorted(set(paths))


def contiguous_fit_block(
    values: np.ndarray, fraction: float, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    length = min(maximum, max(3, int(np.floor(len(values) * fraction))))
    if length >= len(values):
        return values.copy()
    start = int(rng.integers(0, len(values) - length + 1))
    return values[start : start + length]


def select_features(
    normal: np.ndarray,
    abnormal: np.ndarray,
    names: tuple[str, ...],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    normal_bootstrap = normal[rng.integers(0, len(normal), len(normal))]
    abnormal_bootstrap = abnormal[rng.integers(0, len(abnormal), len(abnormal))]
    scale = np.std(normal_bootstrap, axis=0, ddof=1)
    shift = np.full(normal.shape[1], -np.inf, dtype=float)
    valid = scale > 1e-12
    shift[valid] = np.abs(
        np.median(abnormal_bootstrap[:, valid], axis=0)
        - np.median(normal_bootstrap[:, valid], axis=0)
    ) / scale[valid]
    selected = np.argsort(shift)[::-1][:count]
    if len(selected) < 2 or not np.isfinite(shift[selected]).all():
        raise ValueError("fewer than two nonconstant calibration-selected features")
    summary = [
        {
            "index": int(index),
            "name": names[int(index)],
            "calibration_standardized_median_shift": float(shift[index]),
            "normal_q05": float(np.quantile(normal[:, index], 0.05)),
            "normal_q50": float(np.quantile(normal[:, index], 0.50)),
            "normal_q95": float(np.quantile(normal[:, index], 0.95)),
            "abnormal_calibration_q50": float(np.quantile(abnormal[:, index], 0.50)),
        }
        for index in selected
    ]
    return selected.astype(int), summary


def alarm_metrics(normal_alarm: np.ndarray, abnormal_alarm: np.ndarray) -> dict[str, float]:
    truth = np.r_[
        np.zeros(len(normal_alarm), dtype=bool), np.ones(len(abnormal_alarm), dtype=bool)
    ]
    return binary_alarm_metrics(truth, np.r_[normal_alarm, abnormal_alarm])


def evaluate_static(model: object, normal: np.ndarray, abnormal: np.ndarray) -> dict[str, float]:
    return alarm_metrics(model.predict(normal), model.predict(abnormal))


def choose_regression(
    normal: np.ndarray, feature_indices: np.ndarray, predictor_count: int
) -> dict[str, object]:
    split = min(max(10, int(np.floor(0.7 * len(normal)))), len(normal) - 5)
    selection, validation = normal[:split], normal[split:]
    candidates = []
    for response in feature_indices:
        others = np.asarray([value for value in feature_indices if value != response], dtype=int)
        correlation = np.asarray(
            [abs(np.corrcoef(selection[:, response], selection[:, index])[0, 1]) for index in others]
        )
        correlation[~np.isfinite(correlation)] = -np.inf
        predictors = others[np.argsort(correlation)[::-1][:predictor_count]]
        design = np.column_stack([np.ones(split), selection[:, predictors]])
        coefficients = np.linalg.pinv(design) @ selection[:, response]
        prediction = np.column_stack(
            [np.ones(len(validation)), validation[:, predictors]]
        ) @ coefficients
        denominator = float(np.sum((validation[:, response] - np.mean(validation[:, response])) ** 2))
        r2 = 1.0 - float(np.sum((validation[:, response] - prediction) ** 2)) / denominator if denominator else -np.inf
        residual = validation[:, response] - prediction
        residual_scale = float(np.std(residual, ddof=1))
        normality_p = (
            float(kstest((residual - np.mean(residual)) / residual_scale, "norm").pvalue)
            if residual_scale > 1e-12
            else 0.0
        )
        candidates.append(
            {
                "response": int(response),
                "predictors": predictors,
                "normal_validation_r2": float(r2),
                "residual_normality_ks_pvalue": normality_p,
            }
        )
    return max(candidates, key=lambda row: row["normal_validation_r2"])


def evaluate_episode(
    episode: dict[str, object],
    dataset: str,
    config: dict[str, object],
    seed: int,
    episode_index: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1009 * episode_index)
    protocol = config["protocol"]
    normal_train = np.asarray(episode["normal_train"], dtype=float)
    normal_evaluation = np.asarray(episode["normal_evaluation"], dtype=float)
    abnormal_calibration = np.asarray(episode["abnormal_calibration"], dtype=float)
    abnormal_evaluation = np.asarray(episode["abnormal_evaluation"], dtype=float)
    feature_indices, feature_summary = select_features(
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
    )
    fit = normal_fit[:, feature_indices]
    normal = normal_evaluation[:, feature_indices]
    abnormal = abnormal_evaluation[:, feature_indices]

    baseline = MahalanobisAlarm(quantile=0.99).fit(fit)
    baseline_metrics = evaluate_static(baseline, normal, abnormal)

    convex = ConvexHullNOZAlarm(
        false_alarm_fraction=float(protocol["convex_false_alarm_fraction"])
    ).fit(fit)
    convex_metrics = evaluate_static(convex, normal, abnormal)
    convex_result = {
        "parameters": {
            "false_alarm_fraction": convex.false_alarm_fraction,
            "facets": int(len(convex.equations_)),
        },
        "empirical": convex_metrics,
        "execution_passed": bool(np.isfinite(list(convex_metrics.values())).all()),
        "activation_passed": True,
    }

    search = SearchConeNOZAlarm(
        angular_resolution_degrees=float(protocol["search_cone_angle_degrees"]),
        radial_quantile=float(protocol["search_cone_radial_quantile"]),
    ).fit(fit)
    search_metrics = evaluate_static(search, normal, abnormal)
    search_result = {
        "parameters": {
            "angular_resolution_degrees": search.angular_resolution_degrees,
            "radial_quantile": search.radial_quantile,
            "occupied_cones": int(len(search.cone_keys_)),
        },
        "empirical": search_metrics,
        "execution_passed": bool(np.isfinite(list(search_metrics.values())).all()),
        "activation_passed": bool(len(search.cone_keys_) >= 2),
    }

    maximum_scale = min(int(protocol["atg_maximum_scale"]), len(fit) // 3)
    variation = VariationDirectionAlarm(
        minimum_scale=int(protocol["atg_minimum_scale"]), maximum_scale=maximum_scale
    ).fit(fit)
    variation_normal = variation.predict(normal)
    variation_abnormal = variation.predict(abnormal)
    variation_metrics = alarm_metrics(variation_normal, variation_abnormal)
    variation_result = {
        "parameters": {
            "minimum_scale": variation.minimum_scale,
            "maximum_scale": variation.maximum_scale,
            "allowed_direction_vectors": int(len(variation.allowed_directions_)),
        },
        "empirical": variation_metrics,
        "execution_passed": bool(np.isfinite(list(variation_metrics.values())).all()),
        "activation_passed": bool(len(variation.allowed_directions_) >= 2),
    }

    fit_mean = np.mean(normal_fit, axis=0)
    fit_scale = np.std(normal_fit, axis=0, ddof=1)
    fit_scale[fit_scale <= 1e-12] = 1.0
    standardized_fit = (normal_fit - fit_mean) / fit_scale
    standardized_normal = (normal_evaluation - fit_mean) / fit_scale
    standardized_abnormal = (abnormal_evaluation - fit_mean) / fit_scale
    regression = choose_regression(
        standardized_fit,
        feature_indices,
        int(protocol["pump_predictors"]),
    )
    response = int(regression["response"])
    predictors = np.asarray(regression["predictors"], dtype=int)
    window_length = min(30, max(len(predictors) + 2, len(standardized_fit) // 20))
    pump = BayesianWindowRegressionAlarm(
        window_length=window_length,
        forgetting_factor=0.95,
        ridge=1e-6,
        target_false_alarm_rate=0.05,
        significance=0.05,
    ).fit(standardized_fit[:, predictors], standardized_fit[:, response])
    normal_alarm, normal_frozen = deepcopy(pump).predict_update(
        standardized_normal[:, predictors], standardized_normal[:, response]
    )
    abnormal_alarm, abnormal_frozen = deepcopy(pump).predict_update(
        standardized_abnormal[:, predictors], standardized_abnormal[:, response]
    )
    pump_metrics = alarm_metrics(normal_alarm, abnormal_alarm)
    statistical_gate = bool(
        regression["normal_validation_r2"]
        >= float(protocol["pump_minimum_normal_validation_r2"])
        and regression["residual_normality_ks_pvalue"]
        >= float(protocol["pump_normality_significance"])
    )
    pump_result = {
        "parameters": {
            "window_length": window_length,
            "response": episode["feature_names"][response],
            "predictors": [episode["feature_names"][index] for index in predictors],
        },
        "normal_validation_r2": regression["normal_validation_r2"],
        "residual_normality_ks_pvalue": regression["residual_normality_ks_pvalue"],
        "normal_frozen_fraction": float(np.mean(normal_frozen)),
        "abnormal_frozen_fraction": float(np.mean(abnormal_frozen)),
        "empirical": pump_metrics,
        "execution_passed": bool(np.isfinite(list(pump_metrics.values())).all()),
        "activation_passed": statistical_gate,
        "paper_domain_activation_passed": False,
        "paper_domain_denial": "required electrical-pump differential-pressure/speed/flow semantics are not established",
    }

    return {
        "dataset": dataset,
        "episode_id": episode["id"],
        "selected_features": feature_summary,
        "normal_fit_samples": len(normal_fit),
        "prior": episode["prior"],
        "split_policy": episode["split_policy"],
        "baseline_mahalanobis": baseline_metrics,
        "algorithms": {
            "book_3_1_convex_noz": convex_result,
            "book_3_1_nonconvex_noz": search_result,
            "book_3_2_variation_direction": variation_result,
            "book_3_3_electrical_pump": pump_result,
        },
    }


def generate_condenser_samples(
    count: int, parameters: CondenserParameters, rng: np.random.Generator
) -> np.ndarray:
    # Follow Section 3.4.2.2: draw Pc, T2-T1 and terminal difference
    # inside their declared operating ranges, then solve equations (3.92),
    # (3.93), and (3.100) analytically for Dc.
    pressure = rng.uniform(6.0, 11.0, count)
    temperature_difference = rng.uniform(6.0, 12.0, count)
    terminal_difference = rng.uniform(1.5, 6.0, count)
    steam_temperature = 57.66 * (pressure / 9.8e-3) ** (1.0 / 7.46) - 100.0
    outlet = steam_temperature - terminal_difference
    inlet = outlet - temperature_difference
    chi = np.where(inlet <= 26.7, 0.0969 * (1.0 + 0.15 * inlet), 0.4845)
    temperature_factor = np.where(
        inlet <= 35.0,
        1.0 - (0.52 - 0.0072 * parameters.steam_load) * (35.0 - inlet) / 1000.0,
        1.0 + 0.002 * (inlet - 35.0),
    )
    area = (
        2.0
        * np.pi
        * parameters.outer_diameter
        * parameters.tube_length
        * parameters.tube_count
    )
    coefficient = 3.2865 * (
        8.8
        * parameters.latent_heat
        / (
            1000.0
            * 4.1816
            * np.pi
            * parameters.tube_count
            * parameters.inner_diameter**2.25
            * temperature_difference
        )
    ) ** chi * temperature_factor
    target_exponent = np.log1p(temperature_difference / terminal_difference)
    right = (
        target_exponent
        * parameters.latent_heat
        / (coefficient * area * temperature_difference)
    )
    steam_flow = right ** (1.0 / (chi - 1.0))
    values = np.column_stack([pressure, steam_flow, inlet, outlet])
    reconstructed = CondenserPhysicalModel().predict_pressure(
        steam_flow, inlet, outlet, parameters
    )
    if not np.allclose(reconstructed, pressure, rtol=1e-10, atol=1e-10):
        raise RuntimeError("equation-defined condenser sample reconstruction failed")
    return values


def condenser_validation(seed: int, protocol: dict[str, object]) -> dict[str, object]:
    rng = np.random.default_rng(seed + 7919)
    reference = CondenserParameters(2152.7, 24613.0, 14.6996, 12.8538, 0.0182, 0.0247)
    train_count = int(protocol["condenser_synthetic_train_samples"])
    evaluation_count = int(protocol["condenser_synthetic_evaluation_samples"])
    values = generate_condenser_samples(train_count + 2 * evaluation_count, reference, rng)
    values[:, 0] += rng.normal(0.0, 0.005, len(values))
    train = values[:train_count]
    normal = values[train_count : train_count + evaluation_count]
    abnormal = values[train_count + evaluation_count :].copy()
    abnormal[:, 0] += 0.75
    bounds = [
        (reference.latent_heat * 0.95, reference.latent_heat * 1.05),
        (reference.tube_count * 0.95, reference.tube_count * 1.05),
        (reference.tube_length * 0.95, reference.tube_length * 1.05),
        (reference.steam_load * 0.95, reference.steam_load * 1.05),
        (reference.inner_diameter * 0.95, reference.inner_diameter * 1.05),
        (reference.outer_diameter * 0.95, reference.outer_diameter * 1.05),
    ]
    physical = CondenserPhysicalModel().fit(
        train[:, 0],
        train[:, 1],
        train[:, 2],
        train[:, 3],
        bounds=bounds,
        seed=seed,
        maxiter=int(protocol["condenser_fit_maxiter"]),
    )
    monitor = CondenserNOZAlarm(
        physical.parameters_, angular_resolution_degrees=10.0, residual_quantile=0.99
    ).fit(train)
    normal_alarm = monitor.predict(normal)
    abnormal_alarm = monitor.predict(abnormal)
    metrics = alarm_metrics(normal_alarm, abnormal_alarm)
    rate_bounds = condenser_alarm_rate_bounds(
        normal_alarm, abnormal_alarm, confidence=0.99
    )
    reference_vector = np.asarray(list(asdict(reference).values()), dtype=float)
    fitted_vector = np.asarray(list(asdict(physical.parameters_).values()), dtype=float)
    book_point = CondenserPhysicalModel().predict_pressure(
        [906.81], [36.08], [44.95], reference
    )[0]
    fit_activation = bool(physical.fit_score_ >= 0.95)
    zone_activation = bool(
        rate_bounds.false_alarm.upper <= float(protocol["condenser_far_upper_target"])
        and rate_bounds.missed_alarm.upper <= float(protocol["condenser_mar_upper_target"])
    )
    return {
        "dataset": "book_equation_condenser_synthetic",
        "match": "M3",
        "protocol": "P2-equation-defined, not original industrial raw data",
        "table_3_5_parameters": asdict(reference),
        "fitted_parameters": asdict(physical.parameters_),
        "fit_goodness": physical.fit_score_,
        "maximum_parameter_relative_error": float(
            np.max(np.abs(fitted_vector - reference_vector) / reference_vector)
        ),
        "book_abnormality_1_point_predicted_pressure_kpa": float(book_point),
        "book_abnormality_1_observed_pressure_kpa": 10.79,
        "empirical": metrics,
        "bayesian_rate_bounds_99pct": {
            "false_alarm": asdict(rate_bounds.false_alarm),
            "missed_alarm": asdict(rate_bounds.missed_alarm),
        },
        "execution_passed": bool(np.isfinite(list(metrics.values())).all()),
        "physical_fit_activation_passed": fit_activation,
        "zone_performance_activation_passed": zone_activation,
        "activation_passed": bool(fit_activation and zone_activation),
        "industrial_score_reproduction_passed": False,
        "industrial_score_blocker": "original 300-MW condenser daily process data and parameter ensembles are unavailable",
    }


def named_book_items(seed: int) -> dict[str, object]:
    figure_points = np.asarray(
        [
            [0.0, -0.4], [-0.2, -0.2], [0.0, -0.2], [0.2, -0.2],
            [-0.4, 0.0], [0.4, 0.0], [-0.2, 0.2], [0.2, 0.2], [0.0, 0.4],
        ]
    )
    fitness = convex_hull_fitness_index(
        figure_points, [0.2, 0.2], lower=[-0.5, -0.5], upper=[0.5, 0.5]
    )

    rng = np.random.default_rng(seed + 3571)
    atg_values = np.r_[rng.normal(0.0, 1.0, 299), -np.arange(0, 701) + rng.normal(0.0, 1.0, 701)]
    atg = AdaptiveTimeGradient(20, 100).fit(atg_values[:200])
    gradients, directions, scales = atg.transform(atg_values)
    negative = np.flatnonzero((np.arange(len(directions)) >= 299) & (directions < 0))

    pump_rows = [
        {"kp": 0.74, "max_abnormal": 0.3216, "max_normal": 0.1225, "reported_difference": 0.1902, "fitness": 0.9552},
        {"kp": 0.79, "max_abnormal": 0.3612, "max_normal": 0.1331, "reported_difference": 0.2281, "fitness": 0.9532},
        {"kp": 0.84, "max_abnormal": 0.4245, "max_normal": 0.1503, "reported_difference": 0.2742, "fitness": 0.9504},
        {"kp": 0.89, "max_abnormal": 0.5086, "max_normal": 0.1758, "reported_difference": 0.3328, "fitness": 0.9456},
    ]
    for row in pump_rows:
        row["calculated_difference"] = row["max_abnormal"] - row["max_normal"]
        row["absolute_arithmetic_gap"] = abs(
            row["calculated_difference"] - row["reported_difference"]
        )
    eligible = [row for row in pump_rows if row["fitness"] >= 0.95]
    selected = max(eligible, key=lambda row: row["calculated_difference"])
    vif_rows = {
        "0.006": [0.1184, 0.0487, 0.1010, 0.1371],
        "0.009": [0.0619, 0.0305, 0.0534, 0.0988],
        "0.012": [0.0422, 0.0242, 0.0368, 0.0853],
        "0.015": [0.0330, 0.0212, 0.0290, 0.0790],
    }
    selected_ridge = min(float(key) for key, values in vif_rows.items() if max(values) <= 0.1)
    return {
        "figure_3_2": {
            **asdict(fitness),
            "reference_fitness": 9.0 / 13.0,
            "passed": abs(fitness.fitness - 9.0 / 13.0) <= 1e-12,
        },
        "section_3_2_example_1": {
            "reference_change_sample_one_based": 300,
            "first_negative_direction_sample_one_based": int(negative[0] + 1) if len(negative) else None,
            "final_gradient": float(gradients[-1]),
            "minimum_observed_scale": int(np.min(scales[299:])),
            "change_direction_recovered": bool(len(negative)),
            "exact_parameter_score_blocker": "the current ATG is equation-inspired but does not yet expose the paper's V0/beta0/mu/rho fit",
        },
        "tables_3_2_to_3_4": {
            "table_3_4_rows": pump_rows,
            "selected_kp": selected["kp"],
            "reference_selected_kp": 0.84,
            "selected_alarm_threshold": (selected["max_abnormal"] + selected["max_normal"]) / 2.0,
            "reference_alarm_threshold": 0.2874,
            "selected_ridge": selected_ridge,
            "reference_selected_ridge": 0.009,
            "selection_passed": bool(
                selected["kp"] == 0.84
                and abs((selected["max_abnormal"] + selected["max_normal"]) / 2.0 - 0.2874) <= 1e-12
                and selected_ridge == 0.009
            ),
            "book_arithmetic_discrepancy": "Table 3.4 reports 0.3216-0.1225=0.1902 for kp=0.74; direct subtraction is 0.1991.",
        },
        "blocked_original_items": [
            "Section 3.1.3 CSTR alpha_opt=1.4 degrees and detection at t=1395: original generated input sequence/initial conditions unavailable",
            "Section 3.2 industrial variable-frequency-pump scores: raw industrial data unavailable",
            "Section 3.3 pump Figures 3.33-3.47: raw four-case pump data unavailable",
            "Section 3.4 Tables 3.6-3.7 and 100-day uncertainty validation: raw 300-MW condenser data unavailable",
        ],
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
    result["baseline_mahalanobis"] = {
        "episodes": len(episodes),
        "mean_false_alarm_rate": float(np.mean([row["baseline_mahalanobis"]["false_alarm_rate"] for row in episodes])),
        "mean_missed_alarm_rate": float(np.mean([row["baseline_mahalanobis"]["missed_alarm_rate"] for row in episodes])),
        "mean_f1": float(np.mean([row["baseline_mahalanobis"]["f1"] for row in episodes])),
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
    condenser = condenser_validation(seed, config["protocol"])
    named = named_book_items(seed)
    prior_passed = all(
        row["prior"]["finite"] and len(row["prior"]["constant_features"]) < row["prior"]["features"] - 1
        for rows in episode_results.values()
        for row in rows
    )
    result = {
        "seed": seed,
        "named_book_items": named,
        "datasets": episode_results,
        "aggregate_metrics": aggregates,
        "condenser_synthetic_validation": condenser,
        "prior_gate": {
            "passed": prior_passed,
            "episodes": sum(len(rows) for rows in episode_results.values()),
            "smd10towfgr_denied": True,
            "smd10towfgr_reason": config["datasets"]["smd10towfgr"]["status"],
            "real_condenser_denied": True,
            "real_condenser_reason": config["datasets"]["condenser_industrial"]["status"],
        },
        "engineering_activation": {
            "book_3_1_convex_noz": all(
                aggregate["book_3_1_convex_noz"]["activation_rate"] == 1.0
                for aggregate in aggregates.values()
            ),
            "book_3_1_nonconvex_noz": all(
                aggregate["book_3_1_nonconvex_noz"]["activation_rate"] == 1.0
                for aggregate in aggregates.values()
            ),
            "book_3_2_variation_direction": all(
                aggregate["book_3_2_variation_direction"]["activation_rate"] == 1.0
                for aggregate in aggregates.values()
            ),
            "book_3_3_electrical_pump": any(
                aggregate["book_3_3_electrical_pump"]["activation_rate"] > 0.0
                for aggregate in aggregates.values()
            ),
            "book_3_4_condenser": condenser["activation_passed"],
        },
        "paper_score_closure": {
            algorithm_id: False for algorithm_id in config["validated_algorithms"]
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
                "prior_gate": result["prior_gate"],
                "engineering_activation": result["engineering_activation"],
                "named_items": {
                    "figure_3_2": named["figure_3_2"]["passed"],
                    "pump_table_selection": named["tables_3_2_to_3_4"]["selection_passed"],
                    "atg_change_direction": named["section_3_2_example_1"]["change_direction_recovered"],
                },
                "condenser": {
                    "fit_goodness": condenser["fit_goodness"],
                    "f1": condenser["empirical"]["f1"],
                },
                "aggregate_metrics": aggregates,
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
