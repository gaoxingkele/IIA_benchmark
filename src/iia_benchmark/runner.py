from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from iia_benchmark.config import load_experiment_config, load_json_reference
from iia_benchmark.data import (
    build_fcc_alarm_split,
    build_npp_alarm_split,
    build_tep_five_class_split,
    load_piade_alarm_intervals,
    load_piade_alarm_sequences,
    load_smd_alarm_events,
    build_pronto_fault_window_split,
    load_pronto_merged_csv,
    load_fcc_alarm_runs,
    load_npp_alarm_runs,
    load_tep_five_class_alarm_runs,
    pronto_normal_train_evaluation_masks,
    load_skab_csv,
    load_tep_ascii,
    make_synthetic_alarm_run,
    make_synthetic_causal_alarm_series,
    make_synthetic_floods,
    make_synthetic_multivariate_run,
)
from iia_benchmark.evaluation import (
    binary_alarm_metrics,
    multiclass_classification_metrics,
    sequence_accuracy,
)
from iia_benchmark.models import (
    AlarmToken,
    CASIMClassifier,
    ConEAlarmFloodClassifier,
    ConvexHullNOZAlarm,
    CrossConformalAlarmFloodClassifier,
    CTFHAlarmFloodClassifier,
    EmpiricalNextAlarmPredictor,
    HDAMTemplateMatcher,
    MaximumEntropyNextAlarmPredictor,
    MahalanobisAlarm,
    NormalizedTransferEntropyGraph,
    SearchConeNOZAlarm,
    TransferEntropyRanker,
    accelerated_alarm_alignment,
    charm_closed_alarm_patterns,
    design_alarm,
    criterion_c_alarm_flood_detection,
    representative_alarm_patterns,
    smith_waterman_similarity,
)
from iia_benchmark.visualization import (
    build_alarm_visual_analytics,
    export_alarm_visual_report,
)


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def _load_references(root: Path, experiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    references = {
        field: load_json_reference(_resolve(root, experiment[field]))
        for field in ("system", "dataset", "split", "model", "metrics")
    }
    for field in ("flood_detector", "base_model"):
        if field in experiment:
            references[field] = load_json_reference(_resolve(root, experiment[field]))
    return references


def _range(specification: dict[str, Any]) -> np.ndarray:
    return np.linspace(
        float(specification["start"]),
        float(specification["stop"]),
        int(specification["num"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_evidence(paths: list[Path], root: Path) -> dict[str, Any]:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    combined = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"files": records, "combined_sha256": combined}


def _macro_metrics(per_run: list[dict[str, Any]]) -> dict[str, float]:
    keys = tuple(per_run[0]["metrics"])
    return {
        key: float(np.mean([record["metrics"][key] for record in per_run]))
        for key in keys
    }


def _fit_multivariate_estimator(
    model: dict[str, Any], training: np.ndarray
) -> tuple[object, dict[str, Any]]:
    model_id = model["id"]
    if model_id == "mahalanobis":
        estimator = MahalanobisAlarm(quantile=float(model.get("quantile", 0.99))).fit(
            training
        )
        diagnostics = {"threshold": estimator.threshold_}
    elif model_id == "search_cone_noz":
        estimator = SearchConeNOZAlarm(**model.get("parameters", {})).fit(training)
        diagnostics = {
            "search_cones": len(estimator.cone_radii_),
            "angular_resolution_degrees": estimator.angular_resolution_degrees,
        }
    elif model_id == "convex_hull_noz":
        estimator = ConvexHullNOZAlarm(
            false_alarm_fraction=float(model.get("false_alarm_fraction", 0.01))
        ).fit(training)
        diagnostics = {"hull_facets": len(estimator.equations_)}
    else:
        raise ValueError(f"unsupported multivariate estimator config: {model_id}")
    return estimator, diagnostics


def _run_univariate(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    run = make_synthetic_alarm_run(**dataset.get("generator", {}))
    values = run.values[:, int(dataset.get("value_column", 0))]
    search = model["search"]
    result = design_alarm(
        values,
        run.abnormal,
        thresholds=_range(search["thresholds"]),
        delays=search["delays"],
        deadbands=search["deadbands"],
        direction=model.get("direction", "high"),
        targets=tuple(model.get("targets", [0.05, 0.05, 10.0])),
        weights=tuple(model.get("weights", [1.0, 1.0, 0.25])),
    )
    alarm = result.model.predict(values)
    return {
        "parameters": {
            "threshold": result.model.threshold,
            "delay": result.model.delay,
            "deadband": result.model.deadband,
            "direction": result.model.direction,
        },
        "design_loss": result.loss,
        "metrics": binary_alarm_metrics(run.abnormal, alarm),
        "samples": len(values),
        "warning": "Smoke experiment tunes and evaluates on one synthetic run; it is not a leaderboard result.",
    }


def _run_flood_similarity(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episodes = make_synthetic_floods(**references["dataset"].get("generator", {}))
    truth: list[str] = []
    prediction: list[str] = []
    for index, episode in enumerate(episodes):
        candidates = [candidate for j, candidate in enumerate(episodes) if j != index]
        nearest = max(
            candidates,
            key=lambda candidate: smith_waterman_similarity(episode.tags(), candidate.tags()),
        )
        truth.append(episode.label or "unknown")
        prediction.append(nearest.label or "unknown")
    return {
        "metrics": {"nearest_neighbor_accuracy": sequence_accuracy(truth, prediction)},
        "episodes": len(episodes),
        "warning": "Synthetic smoke corpus; use run-group and open-set splits for reported results.",
    }


def _run_multivariate_noz(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    run = make_synthetic_multivariate_run(**dataset.get("generator", {}))
    estimator = ConvexHullNOZAlarm(
        false_alarm_fraction=float(model.get("false_alarm_fraction", 0.01))
    ).fit(run.values[~run.abnormal])
    alarm = estimator.predict(run.values)
    return {
        "metrics": binary_alarm_metrics(run.abnormal, alarm),
        "hull_facets": int(len(estimator.equations_)),
        "samples": len(run.timestamps),
        "warning": "Synthetic smoke corpus; use run-grouped held-out trajectories for reporting.",
    }


def _run_root_cause(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    series = make_synthetic_causal_alarm_series(**dataset.get("generator", {}))
    ranking = TransferEntropyRanker(
        max_lag=int(model.get("max_lag", 10)),
        permutations=int(model.get("permutations", 99)),
        significance=float(model.get("significance", 0.01)),
        seed=int(model.get("seed", 0)),
    ).rank(series, target="TARGET")
    return {
        "ranking": [
            {"tag": tag, "transfer_entropy": score, "lag": lag, "threshold": threshold}
            for tag, score, lag, threshold in ranking
        ],
        "top1_correct": bool(ranking and ranking[0][0] == "ROOT"),
        "warning": "Synthetic binary series; real reports require occurrence counts and surrogate policy.",
    }


def _run_real_multivariate(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    loader = dataset["loader"]
    test_globs = dataset.get("test_globs", [dataset["test_glob"]])
    excluded_stems = set(dataset.get("exclude_test_stems", []))
    test_paths = sorted(
        {
            path
            for pattern in test_globs
            for path in root.glob(pattern)
            if path.stem not in excluded_stems
        }
    )
    if loader == "skab":
        train_path = _resolve(root, dataset["train_path"])
        train = load_skab_csv(train_path)
        tests = [load_skab_csv(path) for path in test_paths]
        label_policy = "native point labels; no point adjustment"
    elif loader == "tep_ascii":
        train_path = _resolve(root, dataset["train_path"])
        fault_start = int(dataset["fault_start"])
        train = load_tep_ascii(
            train_path, sample_period=float(dataset.get("sample_period", 180.0))
        )
        tests = [
            load_tep_ascii(
                path,
                fault_start=fault_start,
                root_cause=path.stem.split("_")[0].upper(),
                sample_period=float(dataset.get("sample_period", 180.0)),
            )
            for path in test_paths
        ]
        label_policy = f"fault onset fixed at zero-based sample {fault_start}"
    else:
        raise ValueError(f"Unsupported real multivariate loader: {loader}")
    if not tests:
        raise ValueError("real multivariate experiment selected no test runs")
    estimator, model_diagnostics = _fit_multivariate_estimator(model, train.values)
    per_run = []
    for path, run in zip(test_paths, tests, strict=True):
        prediction = estimator.predict(run.values)
        per_run.append(
            {
                "run_id": path.relative_to(root).as_posix(),
                "samples": len(run.timestamps),
                "abnormal_samples": int(run.abnormal.sum()),
                "metrics": binary_alarm_metrics(run.abnormal, prediction),
            }
        )
    return {
        "metrics": _macro_metrics(per_run),
        "runs": per_run,
        "train_samples": len(train.timestamps),
        "test_runs": len(tests),
        "model_diagnostics": model_diagnostics,
        "label_policy": label_policy,
        "data_evidence": _data_evidence([train_path, *test_paths], root),
        "reporting_status": "engineering validation; not a leaderboard claim",
    }


def _run_real_next_alarm(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split = references["split"]
    model = references["model"]
    source = _resolve(root, dataset["path"])
    by_equipment = load_piade_alarm_sequences(
        source, window_seconds=float(dataset.get("window_seconds", 86_400.0))
    )
    train_sequences: list[tuple[str, ...]] = []
    test_sequences: list[tuple[str, ...]] = []
    groups: list[dict[str, Any]] = []
    train_fraction = float(split["train_fraction"])
    for equipment, sequences in by_equipment.items():
        boundary = min(max(1, int(len(sequences) * train_fraction)), len(sequences) - 1)
        train = sequences[:boundary]
        test = sequences[boundary:]
        train_sequences.extend(tuple(event.tag for event in sequence) for sequence in train)
        test_sequences.extend(tuple(event.tag for event in sequence) for sequence in test)
        groups.append(
            {
                "equipment_id": equipment,
                "train_windows": len(train),
                "test_windows": len(test),
            }
        )
    predictor = EmpiricalNextAlarmPredictor(
        distance_scale=float(model.get("distance_scale", 3.0))
    ).fit(train_sequences)
    truth: list[str] = []
    predicted: list[str] = []
    top3_hits = 0
    eligible = 0
    total = 0
    for sequence in test_sequences:
        distinct = tuple(dict.fromkeys(sequence))
        for current, target in zip(distinct, distinct[1:]):
            total += 1
            if current not in predictor.vocabulary_ or target not in predictor.vocabulary_:
                continue
            probabilities = predictor.predict_proba((current,))
            if not probabilities:
                continue
            ranking = sorted(probabilities, key=probabilities.get, reverse=True)
            truth.append(target)
            predicted.append(ranking[0])
            top3_hits += int(target in ranking[:3])
            eligible += 1
    if not eligible:
        raise ValueError("PIADE test split contains no evaluable next-alarm transitions")
    return {
        "metrics": {
            "top1_accuracy": sequence_accuracy(truth, predicted),
            "top3_accuracy": top3_hits / eligible,
            "vocabulary_coverage": eligible / total if total else 0.0,
        },
        "train_windows": len(train_sequences),
        "test_windows": len(test_sequences),
        "evaluated_transitions": eligible,
        "candidate_transitions": total,
        "equipment_splits": groups,
        "target_policy": "next distinct alarm within fixed chronological window",
        "data_evidence": _data_evidence([source], root),
        "reporting_status": "engineering validation; not a leaderboard claim",
    }


def _run_real_causal_graph(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    train_path = _resolve(root, dataset["train_path"])
    train = load_tep_ascii(
        train_path, sample_period=float(dataset.get("sample_period", 180.0))
    )
    low_quantile, high_quantile = map(
        float, dataset.get("alarm_quantiles", [0.005, 0.995])
    )
    lower = np.quantile(train.values, low_quantile, axis=0)
    upper = np.quantile(train.values, high_quantile, axis=0)
    analysis_paths = [_resolve(root, value) for value in dataset["analysis_runs"]]
    run_results = []
    for run_index, path in enumerate(analysis_paths):
        run = load_tep_ascii(
            path,
            fault_start=int(dataset["fault_start"]),
            root_cause=path.stem.split("_")[0].upper(),
            sample_period=float(dataset.get("sample_period", 180.0)),
        )
        alarms = ((run.values < lower) | (run.values > upper)).astype(np.int8)
        counts = alarms.sum(axis=0)
        eligible = np.flatnonzero(counts >= int(model["minimum_occurrences"]))
        selected = eligible[
            np.argsort(counts[eligible])[-int(model.get("max_features", 8)) :]
        ]
        series = {run.feature_names[index]: alarms[:, index] for index in selected}
        graph = NormalizedTransferEntropyGraph(
            max_lag=int(model["max_lag"]),
            simulations=int(model["simulations"]),
            significance=float(model["significance"]),
            minimum_occurrences=int(model["minimum_occurrences"]),
            seed=int(model.get("seed", 0)) + run_index,
        )
        edges = graph.infer(series)
        outgoing = {
            name: float(
                sum(edge.score for edge in edges if edge.source == name and edge.direct)
            )
            for name in series
        }
        ranking = sorted(outgoing, key=outgoing.get, reverse=True)
        run_results.append(
            {
                "run_id": path.relative_to(root).as_posix(),
                "fault_id": run.root_cause,
                "selected_alarm_variables": [
                    {"tag": run.feature_names[index], "occurrences": int(counts[index])}
                    for index in selected
                ],
                "directed_edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "score": edge.score,
                        "lag": edge.lag,
                        "threshold": edge.threshold,
                        "direct": edge.direct,
                    }
                    for edge in edges
                ],
                "candidate_root_alarm_ranking": [
                    {"tag": name, "outgoing_direct_score": outgoing[name]}
                    for name in ranking
                ],
            }
        )
    return {
        "runs": run_results,
        "analysis_runs": len(run_results),
        "alarm_threshold_policy": {
            "lower_normal_quantile": low_quantile,
            "upper_normal_quantile": high_quantile,
        },
        "data_evidence": _data_evidence([train_path, *analysis_paths], root),
        "reporting_status": (
            "real-data structural validation; candidate alarm-node ranking is not "
            "fault-injection root-cause accuracy"
        ),
    }


def _run_real_alarm_causal_graph(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Run Book 4.1 NTE/NDTE on grouped, task-matched alarm episodes."""

    dataset = references["dataset"]
    split_config = references["split"]
    model = references["model"]
    loader = str(dataset["loader"])
    evidence_paths: list[Path]
    if loader == "fcc_alarm_zip":
        archive = _resolve(root, dataset["alarm_archive"])
        all_runs = load_fcc_alarm_runs(
            archive,
            scenarios=tuple(dataset["scenarios"]),
            run_numbers=tuple(split_config["test_run_numbers"]),
        )
        split_identity = str(split_config["id"])
        evidence_paths = [archive]
    elif loader == "tep_five_class_alarm_zip":
        archive = _resolve(root, dataset["alarm_archive"])
        loaded = load_tep_five_class_alarm_runs(
            archive, disturbances=tuple(dataset["disturbances"])
        )
        split = build_tep_five_class_split(
            loaded,
            train_per_class=int(split_config["train_per_class"]),
            calibration_per_class=int(split_config["calibration_per_class"]),
            test_per_class=int(split_config["test_per_class"]),
            random_state=int(split_config["random_state"]),
            representation=str(dataset.get("alarm_representation", "state")),
        )
        test_ids = set(split.test_run_ids)
        all_runs = tuple(run for run in loaded if run.run_id in test_ids)
        split_identity = f"{split_config['id']}:{split.random_state}"
        evidence_paths = [archive]
    elif loader == "npp_alarm_extracted_alpha_slice":
        extracted_root = _resolve(root, dataset["extracted_root"])
        archive = _resolve(root, dataset["source_archive"])
        loaded = load_npp_alarm_runs(
            extracted_root,
            alpha=float(dataset["alpha"]),
            fault_families=tuple(dataset["fault_families"]),
            minimum_samples=int(dataset["minimum_samples"]),
            horizon_samples=int(dataset["horizon_samples"]),
            include_normal=bool(dataset.get("include_normal", False)),
        )
        split = build_npp_alarm_split(
            loaded,
            train_per_class=int(split_config["train_per_class"]),
            calibration_per_class=int(split_config["calibration_per_class"]),
            test_per_class=int(split_config["test_per_class"]),
            random_state=int(split_config["random_state"]),
            representation=str(dataset.get("alarm_representation", "state")),
        )
        test_ids = set(split.test_run_ids)
        all_runs = tuple(run for run in loaded if run.run_id in test_ids)
        split_identity = f"{split_config['id']}:{split.random_state}"
        evidence_paths = [archive]
    else:
        raise ValueError(f"Unsupported grouped alarm graph loader: {loader}")

    def label_of(run: object) -> str:
        for name in ("scenario", "disturbance", "fault_family"):
            value = getattr(run, name, None)
            if value is not None:
                return str(value)
        raise ValueError("alarm graph run has no class label")

    per_class = int(model["analysis_runs_per_class"])
    selected_runs = []
    for label in sorted({label_of(run) for run in all_runs}):
        class_runs = sorted(
            (run for run in all_runs if label_of(run) == label),
            key=lambda run: (
                int(getattr(run, "run_number", getattr(run, "sample_number", 0))),
                str(run.run_id),
            ),
        )
        if len(class_runs) < per_class:
            raise ValueError(
                f"{label} has {len(class_runs)} evaluation runs; {per_class} required"
            )
        selected_runs.extend(class_runs[:per_class])

    minimum_ones = int(model["minimum_occurrences"])
    minimum_zeros = int(model.get("minimum_clear_samples", minimum_ones))
    max_features = int(model["max_features"])
    run_results = []
    edge_sets: dict[str, set[tuple[str, str]]] = {}
    labels_by_run: dict[str, str] = {}
    for run_index, run in enumerate(selected_runs):
        matrix = np.asarray(
            run.representation(str(dataset.get("alarm_representation", "state"))),
            dtype=np.int8,
        )
        names = tuple(run.alarm_names)
        ones = matrix.sum(axis=0)
        zeros = len(matrix) - ones
        eligible = np.flatnonzero((ones >= minimum_ones) & (zeros >= minimum_zeros))
        probability = ones / len(matrix)
        entropy = np.zeros_like(probability, dtype=float)
        interior = (probability > 0) & (probability < 1)
        entropy[interior] = -(
            probability[interior] * np.log2(probability[interior])
            + (1.0 - probability[interior])
            * np.log2(1.0 - probability[interior])
        )
        selected = sorted(
            map(int, eligible), key=lambda index: (-entropy[index], names[index])
        )[:max_features]
        series = {names[index]: matrix[:, index] for index in selected}
        edges = []
        if len(series) >= 2:
            graph = NormalizedTransferEntropyGraph(
                max_lag=int(model["max_lag"]),
                simulations=int(model["simulations"]),
                significance=float(model["significance"]),
                minimum_occurrences=minimum_ones,
                seed=int(model.get("seed", 0)) + run_index,
            )
            edges = graph.infer(series)
        direct_edges = {
            (edge.source, edge.target) for edge in edges if edge.direct
        }
        run_id = str(run.run_id)
        label = label_of(run)
        edge_sets[run_id] = direct_edges
        labels_by_run[run_id] = label
        outgoing = {
            name: float(
                sum(edge.score for edge in edges if edge.direct and edge.source == name)
            )
            for name in series
        }
        ranking = sorted(outgoing, key=lambda name: (-outgoing[name], name))
        run_results.append(
            {
                "run_id": run_id,
                "class_label": label,
                "samples": len(matrix),
                "eligible_alarm_variables": len(eligible),
                "selected_alarm_variables": [
                    {
                        "tag": names[index],
                        "ones": int(ones[index]),
                        "zeros": int(zeros[index]),
                        "binary_entropy": float(entropy[index]),
                    }
                    for index in selected
                ],
                "significant_nte_edges": len(edges),
                "direct_ndte_edges": sum(edge.direct for edge in edges),
                "indirect_edges_pruned": sum(not edge.direct for edge in edges),
                "directed_edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "score": edge.score,
                        "lag": edge.lag,
                        "threshold": edge.threshold,
                        "direct": edge.direct,
                    }
                    for edge in edges
                ],
                "candidate_root_alarm_ranking": [
                    {"tag": name, "outgoing_direct_score": outgoing[name]}
                    for name in ranking
                ],
            }
        )

    within, cross = [], []
    run_ids = sorted(edge_sets)
    for left_index, left in enumerate(run_ids):
        for right in run_ids[left_index + 1 :]:
            union = edge_sets[left] | edge_sets[right]
            if not union:
                continue
            score = len(edge_sets[left] & edge_sets[right]) / len(union)
            target = within if labels_by_run[left] == labels_by_run[right] else cross
            target.append(float(score))
    significant = sum(row["significant_nte_edges"] for row in run_results)
    direct = sum(row["direct_ndte_edges"] for row in run_results)
    pruned = sum(row["indirect_edges_pruned"] for row in run_results)
    lags = Counter(
        edge["lag"] for row in run_results for edge in row["directed_edges"]
    )
    activation = {
        "passed": bool(significant and pruned),
        "beacon": "at least one surrogate-significant NTE edge and one NDTE-pruned indirect edge",
        "nte_significant_edges": significant,
        "ndte_pruned_edges": pruned,
        "runs_with_nte_edges": sum(
            row["significant_nte_edges"] > 0 for row in run_results
        ),
        "analysis_runs": len(run_results),
    }
    return {
        "metrics": {
            "graph_activation_rate": activation["runs_with_nte_edges"]
            / len(run_results),
            "mean_significant_edges_per_run": significant / len(run_results),
            "direct_edge_fraction": direct / significant if significant else 0.0,
            "indirect_pruning_fraction": pruned / significant if significant else 0.0,
            "within_class_direct_edge_jaccard": (
                float(np.mean(within)) if within else None
            ),
            "cross_class_direct_edge_jaccard": (
                float(np.mean(cross)) if cross else None
            ),
        },
        "activation": activation,
        "runs": run_results,
        "analysis_runs_per_class": per_class,
        "class_count": len({row["class_label"] for row in run_results}),
        "selection_policy": {
            "partition": "test",
            "split_identity": split_identity,
            "run_choice": "lowest source run/sample numbers in the frozen test partition",
            "alarm_variable_choice": "highest binary entropy after occurrence and clearance guards",
            "minimum_occurrences": minimum_ones,
            "minimum_clear_samples": minimum_zeros,
            "max_features": max_features,
        },
        "lag_histogram": {str(key): value for key, value in sorted(lags.items())},
        "stability_pairs": {
            "within_class_evaluable": len(within),
            "cross_class_evaluable": len(cross),
        },
        "data_evidence": _data_evidence(evidence_paths, root),
        "reporting_status": (
            "Book Chapter 4.1 structural transfer validation; alarm-tag causal "
            "ground truth is unavailable, so no root-cause top-k accuracy is claimed"
        ),
        "limitations": [
            "Fault-family labels group stability comparisons but do not identify a root alarm tag.",
            "Feature selection uses binary entropy inside each fixed evaluation episode and is descriptive, not trained.",
            "The unavailable industrial source records prevent exact paper-table reproduction.",
        ],
    }


def _run_real_visual_analytics(
    root: Path,
    run_dir: Path,
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset = references["dataset"]
    source = _resolve(root, dataset["path"])
    events = load_piade_alarm_intervals(
        source, equipment_id=str(dataset["equipment_id"])
    )
    parameters = dataset.get("visualization", {})
    max_duration_days = float(parameters.get("max_duration_days", 0.0))
    if max_duration_days > 0 and events:
        cutoff = events[0].timestamp + max_duration_days * 86_400.0
        events = tuple(event for event in events if event.timestamp <= cutoff)
    report = build_alarm_visual_analytics(
        events,
        console=str(dataset["equipment_id"]),
        window_seconds=float(parameters.get("window_seconds", 600.0)),
        sample_seconds=float(parameters.get("sample_seconds", 60.0)),
        flood_start=int(parameters.get("flood_start", 10)),
        flood_end=int(parameters.get("flood_end", 5)),
        top_n=int(parameters.get("top_n", 20)),
    )
    html_path, json_path = export_alarm_visual_report(report, run_dir)
    payload = report.as_dict()
    return {
        "events": len(events),
        "activations": int(payload["source"]["activation_count"]),
        "unique_alarm_tags": int(payload["performance"]["unique_alarm_tags"]),
        "flood_intervals": len(payload["burst_plot"]["flood_intervals"]),
        "time_span_days": (
            (events[-1].timestamp - events[0].timestamp) / 86_400.0 if events else 0.0
        ),
        "slice_policy": (
            f"first {max_duration_days:g} days for bounded engineering artifact"
            if max_duration_days > 0
            else "full equipment timeline"
        ),
        "artifacts": {
            "html": html_path.relative_to(root).as_posix(),
            "facts": json_path.relative_to(root).as_posix(),
        },
        "artifact_evidence": _data_evidence([html_path, json_path], root),
        "data_evidence": _data_evidence([source], root),
        "reporting_status": "descriptive real-data validation; not a classifier score",
    }


def _pronto_prefix_lengths(
    split_config: dict[str, Any], episode_length: int, *, minimum: int = 1
) -> tuple[int, ...]:
    requested = split_config.get("prefix_lengths_samples", [episode_length])
    lengths = tuple(
        sorted(
            {
                int(value)
                for value in requested
                if minimum <= int(value) <= episode_length
            }
        )
    )
    if not lengths or lengths[-1] != episode_length:
        lengths = (*lengths, episode_length)
    return lengths


def _validation_parameters(model_config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(
        model_config.get("validation_parameters", model_config.get("parameters", {}))
    )
    if "alphas" in parameters:
        parameters["alphas"] = tuple(float(value) for value in parameters["alphas"])
    return parameters


def _fit_pronto_point_classifier(
    model_config: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> tuple[object, dict[str, Any]]:
    model_id = str(model_config["id"])
    parameters = _validation_parameters(model_config)
    if model_id == "casim":
        classifier = CASIMClassifier(**parameters).fit(X, y)
        diagnostics = {
            "ensemble_classifiers": len(classifier.classifiers_),
            "features_per_classifier": classifier.n_features,
            "loop_training_samples": len(classifier.loop_training_labels_),
        }
    elif model_id == "ctfh_fingerprinting":
        classifier = CTFHAlarmFloodClassifier(**parameters).fit(X, y)
        diagnostics = {
            "profiles": [
                {
                    "label": str(profile.label),
                    "sample_count": profile.sample_count,
                    "consensus_hashes": len(profile.hashes),
                    "variability_index": profile.variability_index,
                }
                for profile in classifier.profiles_
            ]
        }
    elif model_id == "structured_hdam":
        classifier = HDAMTemplateMatcher(**parameters).fit(X, y)
        diagnostics = {
            "validation_parameters": parameters,
            "templates": [
                {
                    "label": str(template.label),
                    "sample_count": template.sample_count,
                    "width": int(template.values.shape[1]),
                    "stability": template.stability,
                }
                for template in classifier.templates_
            ],
        }
    else:
        raise ValueError(f"unsupported PRONTO point classifier config: {model_id}")
    return classifier, diagnostics


def _point_prefix_metrics(
    classifier: object,
    X: np.ndarray,
    y: np.ndarray,
    prefix_lengths: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    method = getattr(classifier, "predict_evolution", None)
    if method is None:
        return {}
    evolution = method(X, list(prefix_lengths))
    return {
        str(length): multiclass_classification_metrics(
            y.tolist(), np.asarray(predictions).tolist()
        )
        for length, predictions in evolution.items()
    }


def _stratified_fit_calibration_indices(
    labels: np.ndarray, calibration_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    values = np.asarray(labels)
    fit_indices: list[int] = []
    calibration_indices: list[int] = []
    for label in sorted(set(values.tolist()), key=repr):
        indices = np.flatnonzero(values == label)
        calibration_count = min(
            len(indices) - 1, max(1, int(np.floor(len(indices) * calibration_fraction)))
        )
        fit_indices.extend(indices[:-calibration_count].tolist())
        calibration_indices.extend(indices[-calibration_count:].tolist())
    return np.asarray(sorted(fit_indices)), np.asarray(sorted(calibration_indices))


def _stratified_fold_ids(labels: np.ndarray, folds: int) -> np.ndarray:
    values = np.asarray(labels)
    if folds < 2:
        raise ValueError("cross-conformal validation requires at least two folds")
    assignments = np.empty(len(values), dtype=int)
    for label in sorted(set(values.tolist()), key=repr):
        indices = np.flatnonzero(values == label)
        if len(indices) < folds:
            raise ValueError(f"class {label!r} has fewer samples than folds")
        assignments[indices] = np.arange(len(indices)) % folds
    return assignments


def _set_metrics_by_prefix(
    classifier: object,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    evolution = classifier.evaluate_evolution(X, y)
    return {
        str(length): {
            key: float(value) if np.isfinite(value) else None
            for key, value in asdict(metrics).items()
        }
        for length, metrics in evolution.items()
    }


def _run_pronto_conformal_classifier(
    model_config: dict[str, Any],
    base_model_config: dict[str, Any],
    split_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    model_id = str(model_config["id"])
    parameters = _validation_parameters(model_config)
    prefix_lengths = _pronto_prefix_lengths(
        split_config,
        X_train.shape[2],
        minimum=int(_validation_parameters(base_model_config).get("window_size", 1)),
    )
    if model_id == "cone_afc":
        calibration_fraction = float(
            split_config.get("calibration_fraction_of_training", 0.4)
        )
        fit_indices, calibration_indices = _stratified_fit_calibration_indices(
            y_train, calibration_fraction
        )
        models = {
            length: _fit_pronto_point_classifier(
                base_model_config,
                X_train[fit_indices, :, :length],
                y_train[fit_indices],
            )[0]
            for length in prefix_lengths
        }
        classifier = ConEAlarmFloodClassifier(
            models,
            error_rate=float(parameters["error_rate"]),
            score_kind=str(parameters.get("score_kind", "probability")),
        ).calibrate(X_train[calibration_indices], y_train[calibration_indices])
        prefix_metrics = _set_metrics_by_prefix(classifier, X_test, y_test)
        diagnostics = {
            "fit_windows": len(fit_indices),
            "calibration_windows": len(calibration_indices),
            "fit_class_counts": dict(
                sorted(Counter(y_train[fit_indices].tolist()).items())
            ),
            "calibration_class_counts": dict(
                sorted(Counter(y_train[calibration_indices].tolist()).items())
            ),
            "thresholds": classifier.thresholds_.values.tolist(),
            "calibration_counts": classifier.thresholds_.calibration_counts.tolist(),
        }
    elif model_id == "cross_conformal_afc":
        fold_count = int(parameters.get("n_folds", 5))
        fold_ids = _stratified_fold_ids(y_train, fold_count)
        models = {
            length: {
                fold: _fit_pronto_point_classifier(
                    base_model_config,
                    X_train[fold_ids != fold, :, :length],
                    y_train[fold_ids != fold],
                )[0]
                for fold in range(fold_count)
            }
            for length in prefix_lengths
        }
        classifier = CrossConformalAlarmFloodClassifier(
            models,
            error_rate=float(parameters["error_rate"]),
            score_kind=str(parameters.get("score_kind", "probability")),
            class_conditional=bool(parameters.get("class_conditional", True)),
            empty_set_policy=str(parameters.get("empty_set_policy", "top_p_value")),
        ).calibrate(X_train, y_train, fold_ids)
        prefix_metrics = _set_metrics_by_prefix(classifier, X_test, y_test)
        diagnostics = {
            "folds": fold_count,
            "fold_class_counts": {
                str(fold): dict(
                    sorted(Counter(y_train[fold_ids == fold].tolist()).items())
                )
                for fold in range(fold_count)
            },
            "calibration_counts": (
                classifier.calibrator_.diagnostics_.calibration_counts.tolist()
            ),
        }
    else:
        raise ValueError(f"unsupported conformal PRONTO classifier: {model_id}")
    return {
        "model_id": model_id,
        "base_model_id": str(base_model_config["id"]),
        "metrics": prefix_metrics[str(prefix_lengths[-1])],
        "prefix_metrics": prefix_metrics,
        "prefix_lengths_samples": list(prefix_lengths),
        "model_diagnostics": diagnostics,
        "metric_kind": "conformal_prediction_set",
    }


def _run_fcc_conformal_classifier(
    model_config: dict[str, Any],
    base_model_config: dict[str, Any],
    split_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate conformal AFC with an explicit complete-run calibration split."""

    model_id = str(model_config["id"])
    parameters = _validation_parameters(model_config)
    prefix_lengths = _pronto_prefix_lengths(
        split_config,
        X_train.shape[2],
        minimum=int(_validation_parameters(base_model_config).get("window_size", 1)),
    )
    if model_id == "cone_afc":
        models = {
            length: _fit_pronto_point_classifier(
                base_model_config, X_train[:, :, :length], y_train
            )[0]
            for length in prefix_lengths
        }
        classifier = ConEAlarmFloodClassifier(
            models,
            error_rate=float(parameters["error_rate"]),
            score_kind=str(parameters.get("score_kind", "probability")),
        ).calibrate(X_calibration, y_calibration)
        prefix_metrics = _set_metrics_by_prefix(classifier, X_test, y_test)
        diagnostics = {
            "fit_runs": len(y_train),
            "calibration_runs": len(y_calibration),
            "fit_class_counts": dict(sorted(Counter(y_train.tolist()).items())),
            "calibration_class_counts": dict(
                sorted(Counter(y_calibration.tolist()).items())
            ),
            "thresholds": classifier.thresholds_.values.tolist(),
            "calibration_counts": classifier.thresholds_.calibration_counts.tolist(),
        }
    elif model_id == "cross_conformal_afc":
        X_pool = np.concatenate((X_train, X_calibration), axis=0)
        y_pool = np.concatenate((y_train, y_calibration), axis=0)
        fold_count = int(parameters.get("n_folds", 5))
        fold_ids = _stratified_fold_ids(y_pool, fold_count)
        models = {
            length: {
                fold: _fit_pronto_point_classifier(
                    base_model_config,
                    X_pool[fold_ids != fold, :, :length],
                    y_pool[fold_ids != fold],
                )[0]
                for fold in range(fold_count)
            }
            for length in prefix_lengths
        }
        classifier = CrossConformalAlarmFloodClassifier(
            models,
            error_rate=float(parameters["error_rate"]),
            score_kind=str(parameters.get("score_kind", "probability")),
            class_conditional=bool(parameters.get("class_conditional", True)),
            empty_set_policy=str(parameters.get("empty_set_policy", "top_p_value")),
        ).calibrate(X_pool, y_pool, fold_ids)
        prefix_metrics = _set_metrics_by_prefix(classifier, X_test, y_test)
        diagnostics = {
            "fit_and_calibration_runs": len(y_pool),
            "folds": fold_count,
            "fold_class_counts": {
                str(fold): dict(sorted(Counter(y_pool[fold_ids == fold].tolist()).items()))
                for fold in range(fold_count)
            },
            "calibration_counts": (
                classifier.calibrator_.diagnostics_.calibration_counts.tolist()
            ),
        }
    else:
        raise ValueError(f"unsupported conformal FCC classifier: {model_id}")
    return {
        "model_id": model_id,
        "base_model_id": str(base_model_config["id"]),
        "metrics": prefix_metrics[str(prefix_lengths[-1])],
        "prefix_metrics": prefix_metrics,
        "prefix_lengths_samples": list(prefix_lengths),
        "model_diagnostics": diagnostics,
        "metric_kind": "conformal_prediction_set",
    }


def _classify_complete_alarm_split(
    model_config: dict[str, Any],
    base_model_config: dict[str, Any] | None,
    split_config: dict[str, Any],
    split: object,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Evaluate one complete-episode alarm split with shared AFC gate semantics."""

    model_id = str(model_config["id"])
    classes = sorted(set(split.y_train.tolist()))
    if model_id in {"cone_afc", "cross_conformal_afc"}:
        if base_model_config is None:
            raise ValueError(f"{model_id} requires a registered base_model")
        classification = _run_fcc_conformal_classifier(
            model_config,
            base_model_config,
            split_config,
            split.X_train,
            split.y_train,
            split.X_calibration,
            split.y_calibration,
            split.X_test,
            split.y_test,
        )
        full_size = len(classes)
        prefix_metrics = classification["prefix_metrics"]
        efficient_prefixes = sum(
            float(row["average_set_size"]) < full_size
            for row in prefix_metrics.values()
        )
        activation = {
            "passed": efficient_prefixes > 0,
            "beacon": "at least one prefix prediction set is smaller than the full label space",
            "efficient_prefixes": efficient_prefixes,
            "prefixes": len(prefix_metrics),
            "class_count": full_size,
        }
        return classes, classification, activation

    X_fit = np.concatenate((split.X_train, split.X_calibration), axis=0)
    y_fit = np.concatenate((split.y_train, split.y_calibration), axis=0)
    classifier, model_diagnostics = _fit_pronto_point_classifier(
        model_config, X_fit, y_fit
    )
    probabilities = classifier.predict_proba(split.X_test)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    prefix_lengths = _pronto_prefix_lengths(
        split_config,
        split.X_train.shape[2],
        minimum=int(_validation_parameters(model_config).get("window_size", 1)),
    )
    classification = {
        "model_id": model_id,
        "metrics": multiclass_classification_metrics(
            split.y_test.tolist(), predictions.tolist()
        ),
        "prefix_metrics": _point_prefix_metrics(
            classifier, split.X_test, split.y_test, prefix_lengths
        ),
        "prefix_lengths_samples": list(prefix_lengths),
        "model_diagnostics": model_diagnostics,
        "metric_kind": "closed_set_point_classification",
    }
    predicted_classes = len(set(predictions.tolist()))
    finite_probabilities = bool(np.isfinite(probabilities).all())
    if model_id == "ctfh_fingerprinting":
        mechanism_value = sum(
            profile["consensus_hashes"] for profile in model_diagnostics["profiles"]
        )
        mechanism_name = "total_consensus_hashes"
        passed = finite_probabilities and predicted_classes >= 2 and mechanism_value > 0
    elif model_id == "structured_hdam":
        mechanism_value = min(
            template["stability"] for template in model_diagnostics["templates"]
        )
        mechanism_name = "minimum_template_stability"
        passed = (
            finite_probabilities
            and predicted_classes >= 2
            and np.isfinite(mechanism_value)
        )
    else:
        mechanism_value = predicted_classes
        mechanism_name = "predicted_classes"
        passed = finite_probabilities and predicted_classes >= 2
    activation = {
        "passed": bool(passed),
        "beacon": "finite scores, at least two predicted classes, and model-specific mechanism activation",
        "predicted_classes": predicted_classes,
        "class_count": len(classes),
        "finite_probabilities": finite_probabilities,
        mechanism_name: float(mechanism_value),
    }
    return classes, classification, activation


def _criterion_c_diagnostics(
    runs: tuple[object, ...] | list[object], parameters: dict[str, int]
) -> list[dict[str, Any]]:
    diagnostics = []
    for run in runs:
        detection = criterion_c_alarm_flood_detection(
            run.alarm_states,
            tag_names=run.alarm_names,
            **parameters,
        )
        transitions = np.maximum(
            detection.delayed_detection
            - np.r_[np.int8(0), detection.delayed_detection[:-1]],
            0,
        )
        diagnostics.append(
            {
                "run_id": run.run_id,
                "maximum_attention_set_cardinality": int(
                    np.max(detection.cardinality, initial=0)
                ),
                "candidate_flood_intervals": int(np.sum(transitions)),
                "candidate_flood_exposure": float(np.mean(detection.delayed_detection)),
            }
        )
    return diagnostics


def _run_real_fcc_alarm_classification(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split_config = references["split"]
    model_config = references["model"]
    alarm_archive = _resolve(root, dataset["alarm_archive"])
    runs = load_fcc_alarm_runs(
        alarm_archive,
        scenarios=tuple(dataset["scenarios"]) if dataset.get("scenarios") else None,
    )
    split = build_fcc_alarm_split(
        runs,
        train_run_numbers=tuple(split_config["train_run_numbers"]),
        calibration_run_numbers=tuple(split_config["calibration_run_numbers"]),
        test_run_numbers=tuple(split_config["test_run_numbers"]),
        representation=str(dataset.get("alarm_representation", "state")),
    )
    classes, classification, activation = _classify_complete_alarm_split(
        model_config,
        references.get("base_model"),
        split_config,
        split,
    )

    detector_parameters = {
        key: int(value)
        for key, value in references["flood_detector"]["parameters"].items()
    }
    criterion_runs = _criterion_c_diagnostics(runs, detector_parameters)

    return {
        **classification,
        "activation": activation,
        "train_runs": len(split.y_train),
        "calibration_runs": len(split.y_calibration),
        "test_runs": len(split.y_test),
        "classes": classes,
        "alarm_tags": list(split.alarm_names),
        "samples_per_run": int(split.X_train.shape[2]),
        "alarm_representation": split.representation,
        "train_class_counts": dict(sorted(Counter(split.y_train.tolist()).items())),
        "calibration_class_counts": dict(
            sorted(Counter(split.y_calibration.tolist()).items())
        ),
        "test_class_counts": dict(sorted(Counter(split.y_test.tolist()).items())),
        "split_policy": "complete run numbers 1-60 train, 61-80 calibration, 81-100 test within every scenario",
        "criterion_c": {
            "parameters": detector_parameters,
            "candidate_intervals": sum(
                row["candidate_flood_intervals"] for row in criterion_runs
            ),
            "runs_with_candidates": sum(
                row["candidate_flood_intervals"] > 0 for row in criterion_runs
            ),
            "per_run": criterion_runs,
            "label_boundary": "criterion-C intervals are candidates, not expert-confirmed floods",
        },
        "data_evidence": _data_evidence([alarm_archive], root),
        "reporting_status": "FCC real-data engineering validation; P1 transfer protocol, not a paper-score reproduction",
        "limitations": [
            "All 16 labels are simulated abnormal situations; no normal-operation class is present.",
            "Scenario labels are used for AFC, while criterion-C intervals remain descriptive candidates.",
            "The fixed run-number split is not the split of any cited AFC paper.",
        ],
    }


def _run_real_tep_alarm_classification(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split_config = references["split"]
    model_config = references["model"]
    alarm_archive = _resolve(root, dataset["alarm_archive"])
    runs = load_tep_five_class_alarm_runs(
        alarm_archive,
        disturbances=(
            tuple(dataset["disturbances"]) if dataset.get("disturbances") else None
        ),
    )
    split = build_tep_five_class_split(
        runs,
        train_per_class=int(split_config["train_per_class"]),
        calibration_per_class=int(split_config["calibration_per_class"]),
        test_per_class=int(split_config["test_per_class"]),
        random_state=int(split_config["random_state"]),
        representation=str(dataset.get("alarm_representation", "state")),
    )
    classes, classification, activation = _classify_complete_alarm_split(
        model_config,
        references.get("base_model"),
        split_config,
        split,
    )
    detector_parameters = {
        key: int(value)
        for key, value in references["flood_detector"]["parameters"].items()
    }
    criterion_runs = _criterion_c_diagnostics(runs, detector_parameters)
    split_ids = {
        "train": split.train_run_ids,
        "calibration": split.calibration_run_ids,
        "test": split.test_run_ids,
    }
    partition_sha256 = hashlib.sha256(
        json.dumps(split_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_by_id = {run.run_id: run for run in runs}
    scaling_by_partition = {
        name: {
            "minimum_scaling_mean": float(
                np.mean([run_by_id[run_id].min_scaling for run_id in run_ids])
            ),
            "maximum_scaling_mean": float(
                np.mean([run_by_id[run_id].max_scaling for run_id in run_ids])
            ),
        }
        for name, run_ids in split_ids.items()
    }
    return {
        **classification,
        "activation": activation,
        "train_runs": len(split.y_train),
        "calibration_runs": len(split.y_calibration),
        "test_runs": len(split.y_test),
        "classes": classes,
        "class_to_label": {
            str(run.class_label): run.disturbance
            for run in sorted(runs, key=lambda item: item.class_label)
        },
        "alarm_tags": list(split.alarm_names),
        "samples_per_run": int(split.X_train.shape[2]),
        "alarm_representation": split.representation,
        "train_class_counts": dict(sorted(Counter(split.y_train.tolist()).items())),
        "calibration_class_counts": dict(
            sorted(Counter(split.y_calibration.tolist()).items())
        ),
        "test_class_counts": dict(sorted(Counter(split.y_test.tolist()).items())),
        "split_policy": (
            "seeded stratified complete-simulation-sample split with 120 train, "
            "40 calibration, and 40 test samples per disturbance"
        ),
        "split_random_state": split.random_state,
        "partition_sha256": partition_sha256,
        "scaling_by_partition": scaling_by_partition,
        "criterion_c": {
            "parameters": detector_parameters,
            "candidate_intervals": sum(
                row["candidate_flood_intervals"] for row in criterion_runs
            ),
            "runs_with_candidates": sum(
                row["candidate_flood_intervals"] > 0 for row in criterion_runs
            ),
            "per_run": criterion_runs,
            "label_boundary": "criterion-C intervals are candidates, not expert-confirmed floods",
        },
        "data_evidence": _data_evidence([alarm_archive], root),
        "reporting_status": (
            "TEP five-class real-data engineering validation; exact public payload "
            "with a locally registered P1 split, not a paper-score reproduction"
        ),
        "limitations": [
            "The five-class payload has no normal-operation class.",
            "Fault labels supervise AFC, while criterion-C intervals remain descriptive candidates.",
            "The seeded 60/20/20 split is not asserted to equal a cited paper split.",
        ],
    }


def _run_real_npp_alarm_classification(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split_config = references["split"]
    model_config = references["model"]
    extracted_root = _resolve(root, dataset["extracted_root"])
    source_archive = _resolve(root, dataset["source_archive"])
    runs = load_npp_alarm_runs(
        extracted_root,
        alpha=float(dataset["alpha"]),
        fault_families=tuple(dataset["fault_families"]),
        minimum_samples=int(dataset["minimum_samples"]),
        horizon_samples=int(dataset["horizon_samples"]),
        include_normal=bool(dataset.get("include_normal", False)),
    )
    split = build_npp_alarm_split(
        runs,
        train_per_class=int(split_config["train_per_class"]),
        calibration_per_class=int(split_config["calibration_per_class"]),
        test_per_class=int(split_config["test_per_class"]),
        random_state=int(split_config["random_state"]),
        representation=str(dataset.get("alarm_representation", "state")),
    )
    classes, classification, activation = _classify_complete_alarm_split(
        model_config,
        references.get("base_model"),
        split_config,
        split,
    )
    detector_parameters = {
        key: int(value)
        for key, value in references["flood_detector"]["parameters"].items()
    }
    criterion_runs = _criterion_c_diagnostics(runs, detector_parameters)
    split_ids = {
        "train": split.train_run_ids,
        "calibration": split.calibration_run_ids,
        "test": split.test_run_ids,
    }
    partition_sha256 = hashlib.sha256(
        json.dumps(split_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **classification,
        "activation": activation,
        "train_runs": len(split.y_train),
        "calibration_runs": len(split.y_calibration),
        "test_runs": len(split.y_test),
        "eligible_runs_before_deduplication": len(runs),
        "unused_or_deduplicated_runs": len(split.unused_run_ids),
        "duplicate_nonrepresentative_runs": len(split.duplicate_run_ids),
        "cross_label_conflicting_runs": len(split.conflicting_run_ids),
        "cross_label_conflicting_run_ids": list(split.conflicting_run_ids),
        "classes": classes,
        "excluded_fault_families": list(dataset.get("excluded_fault_families", [])),
        "alarm_tags": list(split.alarm_names),
        "samples_per_run": int(split.X_train.shape[2]),
        "sample_seconds": float(runs[0].sample_seconds),
        "alarm_representation": split.representation,
        "alpha": float(dataset["alpha"]),
        "train_class_counts": dict(sorted(Counter(split.y_train.tolist()).items())),
        "calibration_class_counts": dict(
            sorted(Counter(split.y_calibration.tolist()).items())
        ),
        "test_class_counts": dict(sorted(Counter(split.y_test.tolist()).items())),
        "split_policy": (
            "alpha=0.50; at least 160 source samples; exact state-or-edge "
            "trajectory components are deduplicated; cross-label conflicts are "
            "removed; one representative per component; 28/10/10 per class"
        ),
        "split_random_state": split.random_state,
        "partition_sha256": partition_sha256,
        "criterion_c": {
            "parameters": detector_parameters,
            "candidate_intervals": sum(
                row["candidate_flood_intervals"] for row in criterion_runs
            ),
            "runs_with_candidates": sum(
                row["candidate_flood_intervals"] > 0 for row in criterion_runs
            ),
            "per_run": criterion_runs,
            "label_boundary": "criterion-C intervals are candidates, not expert-confirmed floods",
        },
        "data_evidence": _data_evidence([source_archive], root),
        "reporting_status": (
            "NPP alpha-0.50 engineering transfer validation; P1 grouped unique-"
            "trajectory split, not a paper-score reproduction"
        ),
        "limitations": [
            "Normal has one run per alpha and is excluded from balanced closed-set scoring.",
            "MD has one unique 160-sample alarm trajectory across 100 runs and is excluded from independent train/test scoring.",
            "Runs shorter than 160 samples and state/edge duplicate or cross-label-conflict components are excluded by G0.",
            "Other alpha slices are reserved for grouped threshold-robustness evaluation.",
        ],
    }


def _fcc_alarm_tokens(run: object) -> tuple[AlarmToken, ...]:
    episode = run.to_episode(include_clearances=False)
    return tuple(
        AlarmToken(event.tag, event.timestamp, event.priority)
        for event in episode.activations()
    )


def _run_real_fcc_book_sequence_method(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split_config = references["split"]
    model_config = references["model"]
    alarm_archive = _resolve(root, dataset["alarm_archive"])
    runs = load_fcc_alarm_runs(
        alarm_archive,
        scenarios=tuple(dataset["scenarios"]) if dataset.get("scenarios") else None,
    )
    train_ids = set(map(int, split_config["train_run_numbers"]))
    calibration_ids = set(map(int, split_config["calibration_run_numbers"]))
    test_ids = set(map(int, split_config["test_run_numbers"]))
    fit_runs = [run for run in runs if run.run_number in train_ids | calibration_ids]
    test_runs = [run for run in runs if run.run_number in test_ids]
    classes = sorted({run.scenario for run in runs})
    model_id = str(model_config["id"])
    parameters = dict(
        model_config.get("validation_parameters", model_config.get("parameters", {}))
    )

    if model_id == "accelerated_alarm_alignment":
        prototypes_per_class = int(parameters.pop("prototypes_per_class", 10))
        prototypes = {
            label: [
                _fcc_alarm_tokens(run)
                for run in fit_runs
                if run.scenario == label
            ][:prototypes_per_class]
            for label in classes
        }
        truth: list[str] = []
        prediction: list[str] = []
        best_scores: list[float] = []
        evaluated_cells = 0
        for run in test_runs:
            query = _fcc_alarm_tokens(run)
            class_scores: dict[str, float] = {}
            for label, candidates in prototypes.items():
                results = [
                    accelerated_alarm_alignment(query, candidate, **parameters)
                    for candidate in candidates
                ]
                evaluated_cells += sum(result.cells_evaluated for result in results)
                class_scores[label] = max(
                    (result.similarity for result in results), default=0.0
                )
            truth.append(run.scenario)
            prediction.append(max(class_scores, key=class_scores.get))
            best_scores.append(max(class_scores.values()))
        metrics = multiclass_classification_metrics(truth, prediction)
        activation = {
            "passed": len(set(prediction)) >= 2 and max(best_scores, default=0.0) > 0,
            "beacon": "positive alignment similarity and at least two predicted classes",
            "predicted_classes": len(set(prediction)),
            "class_count": len(classes),
            "mean_best_similarity": float(np.mean(best_scores)),
            "cells_evaluated": evaluated_cells,
        }
        method_result = {
            "metrics": metrics,
            "prototype_runs_per_class": prototypes_per_class,
            "alignment_parameters": parameters,
            "mean_best_similarity": float(np.mean(best_scores)),
            "cells_evaluated": evaluated_cells,
        }
    elif model_id == "charm_representative_patterns":
        models = {}
        for label in classes:
            transactions = [
                run.to_episode(include_clearances=False).tags(unique=True)
                for run in fit_runs
                if run.scenario == label
            ]
            closed = charm_closed_alarm_patterns(
                transactions,
                minimum_support=float(parameters["minimum_support"]),
            )
            representatives = representative_alarm_patterns(
                closed,
                similarity_threshold=float(parameters["similarity_threshold"]),
            )
            models[label] = {"closed": closed, "representatives": representatives}
        if any(not value["representatives"] for value in models.values()):
            missing = [label for label, value in models.items() if not value["representatives"]]
            raise RuntimeError(f"FCC CHARM produced no representatives for {missing}")

        def jaccard(left: set[str], right: frozenset[str]) -> float:
            union = left | set(right)
            return len(left & set(right)) / len(union) if union else 1.0

        truth = []
        prediction = []
        best_scores = []
        for run in test_runs:
            transaction = set(run.to_episode(include_clearances=False).tags(unique=True))
            scores = {
                label: max(
                    jaccard(transaction, representative.items)
                    for representative in value["representatives"]
                )
                for label, value in models.items()
            }
            truth.append(run.scenario)
            prediction.append(max(scores, key=scores.get))
            best_scores.append(max(scores.values()))
        metrics = multiclass_classification_metrics(truth, prediction)
        pattern_counts = {
            label: {
                "closed_patterns": len(value["closed"]),
                "representative_patterns": len(value["representatives"]),
            }
            for label, value in models.items()
        }
        activation = {
            "passed": len(set(prediction)) >= 2 and all(
                row["representative_patterns"] > 0 for row in pattern_counts.values()
            ),
            "beacon": "every class has a representative pattern and predictions span at least two classes",
            "predicted_classes": len(set(prediction)),
            "class_count": len(classes),
            "total_closed_patterns": sum(
                row["closed_patterns"] for row in pattern_counts.values()
            ),
            "total_representative_patterns": sum(
                row["representative_patterns"] for row in pattern_counts.values()
            ),
        }
        method_result = {
            "metrics": metrics,
            "pattern_counts": pattern_counts,
            "parameters": parameters,
            "mean_best_jaccard": float(np.mean(best_scores)),
        }
    elif model_id == "maximum_entropy_next_alarm":
        fit_sequences = [_fcc_alarm_tokens(run) for run in fit_runs]
        predictor = MaximumEntropyNextAlarmPredictor(**parameters).fit(fit_sequences)
        truth = []
        prediction = []
        top3_hits = 0
        negative_log_likelihood = []
        brier = []
        lead_times = []
        candidate_transitions = 0
        nonuniform_predictions = 0
        vocabulary = tuple(predictor.vocabulary_)
        for run in test_runs:
            sequence = _fcc_alarm_tokens(run)
            for index in range(1, len(sequence)):
                candidate_transitions += 1
                target = sequence[index].tag
                probabilities = predictor.predict_proba(sequence[:index])
                if target not in probabilities or not probabilities:
                    continue
                ranking = sorted(probabilities, key=probabilities.get, reverse=True)
                truth.append(target)
                prediction.append(ranking[0])
                top3_hits += int(target in ranking[:3])
                probability = max(float(probabilities[target]), 1e-15)
                negative_log_likelihood.append(-np.log(probability))
                vector = np.asarray([probabilities.get(tag, 0.0) for tag in vocabulary])
                target_vector = np.asarray([tag == target for tag in vocabulary], dtype=float)
                brier.append(float(np.mean((vector - target_vector) ** 2)))
                lead_times.append(sequence[index].timestamp - sequence[index - 1].timestamp)
                nonuniform_predictions += int(
                    np.ptp(np.asarray(list(probabilities.values()), dtype=float)) > 1e-12
                )
        if not truth:
            raise RuntimeError("FCC maximum-entropy evaluation has no covered transitions")
        top1 = sequence_accuracy(truth, prediction)
        metrics = {
            "top1_accuracy": top1,
            "top3_accuracy": top3_hits / len(truth),
            "negative_log_likelihood": float(np.mean(negative_log_likelihood)),
            "brier_score": float(np.mean(brier)),
            "mean_lead_time_seconds": float(np.mean(lead_times)),
            "vocabulary_coverage": len(truth) / candidate_transitions,
        }
        activation = {
            "passed": nonuniform_predictions > 0 and len(set(prediction)) >= 2,
            "beacon": "non-uniform next-alarm distributions and at least two predicted tags",
            "nonuniform_predictions": nonuniform_predictions,
            "evaluated_transitions": len(truth),
            "candidate_transitions": candidate_transitions,
            "predicted_tags": len(set(prediction)),
            "vocabulary_size": len(vocabulary),
        }
        method_result = {
            "metrics": metrics,
            "learned_weights": predictor.weights_.tolist(),
            "vocabulary_size": len(vocabulary),
            "candidate_transitions": candidate_transitions,
            "evaluated_transitions": len(truth),
            "tie_policy": "same-minute activations retain source alarm-column order",
        }
    else:
        raise ValueError(f"unsupported FCC book sequence method: {model_id}")

    return {
        "model_id": model_id,
        **method_result,
        "activation": activation,
        "fit_runs": len(fit_runs),
        "test_runs": len(test_runs),
        "classes": classes,
        "alarm_representation": "ordered rising-edge activation sequence",
        "split_policy": "complete run numbers 1-80 fit and 81-100 test; registered engineering hyperparameters are fixed before test",
        "data_evidence": _data_evidence([alarm_archive], root),
        "reporting_status": "FCC real-data engineering validation; P1 transfer protocol, not a paper-score reproduction",
        "limitations": [
            "Scenario-conditioned run sequences are used; no expert-confirmed flood interval labels exist.",
            "Simultaneous alarm activations have no sub-minute causal order.",
            "The fixed FCC run split is not the cited paper protocol."
        ],
    }
def _run_real_pronto_alarm_classification(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split_config = references["split"]
    model_config = references["model"]
    paths = [_resolve(root, value) for value in dataset["paths"]]
    runs = [
        load_pronto_merged_csv(
            path,
            alarm_column_count=int(dataset.get("alarm_column_count", 12)),
            label_column=str(dataset.get("label_column", "Fault")),
        )
        for path in paths
    ]
    split = build_pronto_fault_window_split(
        runs,
        window_size=int(split_config["window_size_samples"]),
        train_fraction=float(split_config["train_fraction"]),
        purge_windows=int(split_config.get("purge_windows", 1)),
        excluded_labels=tuple(split_config.get("excluded_labels", ["Normal"])),
        alarm_representation=str(dataset.get("alarm_representation", "state")),
    )
    model_id = str(model_config["id"])
    if model_id in {"cone_afc", "cross_conformal_afc"}:
        classification = _run_pronto_conformal_classifier(
            model_config,
            references["base_model"],
            split_config,
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
        )
    else:
        classifier, model_diagnostics = _fit_pronto_point_classifier(
            model_config, split.X_train, split.y_train
        )
        probabilities = classifier.predict_proba(split.X_test)
        predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
        prefix_lengths = _pronto_prefix_lengths(
            split_config,
            split.X_train.shape[2],
            minimum=int(_validation_parameters(model_config).get("window_size", 1)),
        )
        classification = {
            "model_id": model_id,
            "metrics": multiclass_classification_metrics(
                split.y_test.tolist(), predictions.tolist()
            ),
            "prefix_metrics": _point_prefix_metrics(
                classifier, split.X_test, split.y_test, prefix_lengths
            ),
            "prefix_lengths_samples": list(prefix_lengths),
            "model_diagnostics": model_diagnostics,
            "metric_kind": "closed_set_point_classification",
        }

    detector_parameters = references["flood_detector"]["parameters"]
    criterion_runs = []
    for run in runs:
        detection = criterion_c_alarm_flood_detection(
            run.alarm_states,
            tag_names=run.alarm_names,
            **{key: int(value) for key, value in detector_parameters.items()},
        )
        transitions = np.maximum(
            detection.delayed_detection
            - np.r_[np.int8(0), detection.delayed_detection[:-1]],
            0,
        )
        criterion_runs.append(
            {
                "run_id": run.run_id,
                "evaluated_points": len(detection.sample_indices),
                "maximum_attention_set_cardinality": int(
                    np.max(detection.cardinality, initial=0)
                ),
                "flood_intervals": int(np.sum(transitions)),
                "flood_exposure": float(np.mean(detection.delayed_detection)),
            }
        )

    label_counts = Counter(label for run in runs for label in run.labels.tolist())
    return {
        **classification,
        "train_windows": len(split.y_train),
        "test_windows": len(split.y_test),
        "window_size_samples": int(split.X_train.shape[2]),
        "alarm_tags": list(split.alarm_names),
        "alarm_representation": str(dataset.get("alarm_representation", "state")),
        "train_class_counts": dict(sorted(Counter(split.y_train.tolist()).items())),
        "test_class_counts": dict(sorted(Counter(split.y_test.tolist()).items())),
        "source_label_counts": dict(sorted(label_counts.items())),
        "groups": [asdict(group) for group in split.groups],
        "criterion_c": {
            "parameters": detector_parameters,
            "runs": criterion_runs,
            "confirmed_flood_intervals": sum(
                record["flood_intervals"] for record in criterion_runs
            ),
        },
        "data_evidence": _data_evidence(paths, root),
        "target_policy": (
            "closed-set classification of non-Normal, fixed-length alarm-state windows; "
            "these fault-conditioned windows are not relabelled as confirmed floods"
        ),
        "split_policy": (
            "per contiguous fault segment: earlier non-overlapping windows train, "
            "one or more complete windows purged, later windows test"
        ),
        "reporting_status": (
            "real-data surrogate engineering validation; not leaderboard-eligible and "
            "not a paper-score reproduction"
        ),
        "limitations": [
            "PRONTO provides fault-condition labels rather than expert flood-episode labels.",
            "All classes cannot be separated by test day, so train/test windows may share a source day.",
            (
                "Prefix metrics are engineering curves on fault-conditioned windows; "
                "the paper-specific online protocol is not reproduced."
            ),
            (
                "Conformal coverage in this surrogate run is descriptive and does not "
                "establish the paper's finite-sample guarantee."
            ),
        ],
    }


def _datetime64_seconds(value: str) -> float:
    return float(np.datetime64(value).astype("datetime64[s]").astype(np.int64))


def _run_real_smd_alarm_flood_detection(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split = references["split"]
    model = references["model"]
    source = _resolve(root, dataset["path"])
    by_turbine = load_smd_alarm_events(
        source,
        turbines=dataset.get("turbines"),
        event_types=dataset.get("event_types", ["Alarm log (A)", "Warning log (W)"]),
    )
    bin_seconds = int(split["bin_seconds"])
    if bin_seconds < 1 or 86_400 % bin_seconds:
        raise ValueError("SMD bin_seconds must be a positive divisor of one day")
    start = _datetime64_seconds(str(split["period_start"]))
    stop = _datetime64_seconds(str(split["period_stop"]))
    if stop < start:
        raise ValueError("SMD period_stop precedes period_start")
    days = int((stop - start) // 86_400) + 1
    bins_per_day = 86_400 // bin_seconds
    parameters = model["parameters"]
    criterion_parameters = {
        "attention_window": max(1, int(np.ceil(parameters["attention_window"] / bin_seconds))),
        "long_standing_window": max(
            1, int(np.ceil(parameters["long_standing_window"] / bin_seconds))
        ),
        "update_step": max(1, int(np.ceil(parameters["update_step"] / bin_seconds))),
        "threshold": int(parameters["threshold"]),
        "delay_samples": max(1, int(np.ceil(parameters["delay_samples"] / bin_seconds))),
    }
    daily_records: list[dict[str, Any]] = []
    total_events = 0
    total_unique_occurrences = 0
    total_evaluated = 0
    total_candidate_intervals = 0
    total_candidate_bins = 0
    event_type_counts = {"Alarm log (A)": 0, "Warning log (W)": 0}
    per_turbine: list[dict[str, Any]] = []
    for turbine, events in by_turbine.items():
        tags = tuple(sorted({event.tag for event in events}))
        tag_index = {tag: index for index, tag in enumerate(tags)}
        grouped: dict[int, list[object]] = {}
        for event in events:
            day_index = int((event.timestamp - start) // 86_400)
            if 0 <= day_index < days:
                grouped.setdefault(day_index, []).append(event)
                event_type_counts[
                    "Alarm log (A)" if event.priority == 1 else "Warning log (W)"
                ] += 1
        turbine_maximum = 0
        turbine_intervals = 0
        turbine_candidate_days = 0
        turbine_events = sum(len(values) for values in grouped.values())
        total_events += turbine_events
        empty_day_evaluations = max(
            0,
            len(
                range(
                    criterion_parameters["attention_window"] - 1,
                    bins_per_day,
                    criterion_parameters["update_step"],
                )
            ),
        )
        total_evaluated += (days - len(grouped)) * empty_day_evaluations
        for day_index, event_rows in sorted(grouped.items()):
            occurrences = np.zeros((bins_per_day, len(tags)), dtype=np.int8)
            day_start = start + day_index * 86_400
            for event in event_rows:
                bin_index = min(
                    bins_per_day - 1,
                    max(0, int((event.timestamp - day_start) // bin_seconds)),
                )
                occurrences[bin_index, tag_index[event.tag]] = 1
            unique_occurrences = int(np.sum(occurrences))
            total_unique_occurrences += unique_occurrences
            detection = criterion_c_alarm_flood_detection(
                occurrences,
                tag_names=tags,
                **criterion_parameters,
            )
            starts = int(
                np.sum(
                    (detection.delayed_detection == 1)
                    & (
                        np.r_[0, detection.delayed_detection[:-1]]
                        == 0
                    )
                )
            )
            candidate_bins = int(np.sum(detection.delayed_detection))
            maximum = int(np.max(detection.cardinality, initial=0))
            total_evaluated += len(detection.sample_indices)
            total_candidate_intervals += starts
            total_candidate_bins += candidate_bins
            turbine_maximum = max(turbine_maximum, maximum)
            turbine_intervals += starts
            turbine_candidate_days += int(candidate_bins > 0)
            daily_records.append(
                {
                    "turbine": turbine,
                    "date": str(
                        np.datetime64(str(split["period_start"]))
                        + np.timedelta64(day_index, "D")
                    ),
                    "source_events": len(event_rows),
                    "unique_tag_bins": unique_occurrences,
                    "maximum_attention_cardinality": maximum,
                    "candidate_intervals": starts,
                    "candidate_bins": candidate_bins,
                }
            )
        per_turbine.append(
            {
                "turbine": turbine,
                "events": turbine_events,
                "distinct_codes": len(tags),
                "maximum_attention_cardinality": turbine_maximum,
                "candidate_device_days": turbine_candidate_days,
                "candidate_intervals": turbine_intervals,
            }
        )
    top_dense_days = sorted(
        daily_records,
        key=lambda row: (
            -row["maximum_attention_cardinality"],
            -row["source_events"],
            row["turbine"],
            row["date"],
        ),
    )[:20]
    return {
        "events": total_events,
        "event_type_counts": event_type_counts,
        "turbines": len(by_turbine),
        "device_days": len(by_turbine) * days,
        "nonempty_device_days": len(daily_records),
        "distinct_codes_union": len(
            {event.tag for events in by_turbine.values() for event in events}
        ),
        "unique_tag_bins": total_unique_occurrences,
        "criterion_c_parameters_in_bins": criterion_parameters,
        "candidate_intervals": total_candidate_intervals,
        "candidate_device_days": sum(
            row["candidate_device_days"] for row in per_turbine
        ),
        "candidate_exposure": (
            total_candidate_bins / total_evaluated if total_evaluated else 0.0
        ),
        "maximum_attention_cardinality": max(
            (row["maximum_attention_cardinality"] for row in per_turbine), default=0
        ),
        "per_turbine": per_turbine,
        "top_dense_device_days": top_dense_days,
        "data_evidence": _data_evidence([source], root),
        "event_policy": (
            "Alarm and Warning log rows are timestamped occurrences binned without "
            "inventing activation-clearance state; duplicate code/bin occurrences collapse"
        ),
        "reporting_status": (
            "real event-log descriptive validation; detected intervals are Criterion C "
            "candidates, not expert-confirmed alarm floods or leaderboard labels"
        ),
    }


def _run_real_pronto_multivariate(
    root: Path, references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dataset = references["dataset"]
    split = references["split"]
    model = references["model"]
    paths = [_resolve(root, value) for value in dataset["paths"]]
    runs = [
        load_pronto_merged_csv(
            path,
            alarm_column_count=int(dataset.get("alarm_column_count", 12)),
            label_column=str(dataset.get("label_column", "Fault")),
        )
        for path in paths
    ]
    process_names = runs[0].process_names
    if any(run.process_names != process_names for run in runs):
        raise ValueError("PRONTO test days have inconsistent process columns")
    masks = [
        pronto_normal_train_evaluation_masks(
            run.labels,
            normal_label=str(split.get("normal_label", "Normal")),
            train_fraction=float(split["train_fraction"]),
            purge_samples=int(split.get("purge_samples", 60)),
        )
        for run in runs
    ]
    training = np.vstack(
        [run.process_values[train_mask] for run, (train_mask, _) in zip(runs, masks)]
    )
    estimator, model_diagnostics = _fit_multivariate_estimator(model, training)
    normal_label = str(split.get("normal_label", "Normal"))
    per_run = []
    for path, run, (train_mask, evaluation_mask) in zip(
        paths, runs, masks, strict=True
    ):
        truth = run.labels[evaluation_mask] != normal_label
        prediction = estimator.predict(run.process_values[evaluation_mask])
        per_run.append(
            {
                "run_id": path.relative_to(root).as_posix(),
                "train_normal_samples": int(np.sum(train_mask)),
                "evaluation_samples": int(np.sum(evaluation_mask)),
                "evaluation_normal_samples": int(np.sum(~truth)),
                "evaluation_fault_samples": int(np.sum(truth)),
                "metrics": binary_alarm_metrics(truth, prediction),
            }
        )
    return {
        "metrics": _macro_metrics(per_run),
        "runs": per_run,
        "train_normal_samples": len(training),
        "features": list(process_names),
        "model_diagnostics": model_diagnostics,
        "label_policy": f"Fault != {normal_label!r}; native point labels; no point adjustment",
        "split_policy": (
            "first fraction of every contiguous Normal segment trains; a complete "
            "sample gap is purged; later Normal samples and all fault samples evaluate"
        ),
        "data_evidence": _data_evidence(paths, root),
        "reporting_status": (
            "real-data engineering validation; source-day reuse prevents leaderboard use"
        ),
    }


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    root = path.parents[2]
    experiment = load_experiment_config(path)
    references = _load_references(root, experiment)
    task = experiment["task"]
    run_dir = _resolve(root, experiment["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if task == "univariate_alarm_design":
        result = _run_univariate(references)
    elif task == "alarm_flood_similarity":
        result = _run_flood_similarity(references)
    elif task == "multivariate_noz":
        result = _run_multivariate_noz(references)
    elif task == "root_cause_transfer_entropy":
        result = _run_root_cause(references)
    elif task == "real_multivariate_detection":
        result = _run_real_multivariate(root, references)
    elif task == "real_next_alarm_forecasting":
        result = _run_real_next_alarm(root, references)
    elif task == "real_causal_graph":
        result = _run_real_causal_graph(root, references)
    elif task == "real_alarm_causal_graph":
        result = _run_real_alarm_causal_graph(root, references)
    elif task == "real_visual_analytics":
        result = _run_real_visual_analytics(root, run_dir, references)
    elif task == "real_pronto_alarm_classification":
        result = _run_real_pronto_alarm_classification(root, references)
    elif task == "real_fcc_alarm_classification":
        result = _run_real_fcc_alarm_classification(root, references)
    elif task == "real_tep_alarm_classification":
        result = _run_real_tep_alarm_classification(root, references)
    elif task == "real_npp_alarm_classification":
        result = _run_real_npp_alarm_classification(root, references)
    elif task == "real_fcc_book_sequence_method":
        result = _run_real_fcc_book_sequence_method(root, references)
    elif task == "real_smd_alarm_flood_detection":
        result = _run_real_smd_alarm_flood_detection(root, references)
    elif task == "real_pronto_multivariate_detection":
        result = _run_real_pronto_multivariate(root, references)
    else:
        raise ValueError(f"Unsupported runnable task: {task}")
    payload = {
        "experiment_id": experiment["id"],
        "task": task,
        "result": result,
        "config": str(path.relative_to(root)).replace("\\", "/"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if experiment["outputs"].get("summary"):
        summary_path = _resolve(root, experiment["outputs"]["summary"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if summary_path.suffix.lower() == ".md":
            serialized = (
                f"# {experiment['id']}\n\n"
                "Generated engineering-validation record. Synthetic smoke results are "
                "not leaderboard evidence.\n\n```json\n"
                f"{serialized}\n```\n"
            )
        else:
            serialized += "\n"
        summary_path.write_text(serialized, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an IIA benchmark experiment")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
