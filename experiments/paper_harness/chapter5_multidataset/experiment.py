#!/usr/bin/env python3
"""Book Chapter 5 multi-dataset alarm-flood validation harness."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data.fcc import load_fcc_alarm_runs  # noqa: E402
from iia_benchmark.data.npp_alarm import (  # noqa: E402
    build_npp_alarm_split,
    load_npp_alarm_runs,
)
from iia_benchmark.data.tep_alarm import (  # noqa: E402
    build_tep_five_class_split,
    load_tep_five_class_alarm_runs,
)
from iia_benchmark.evaluation.metrics import (  # noqa: E402
    multiclass_classification_metrics,
)
from iia_benchmark.models import (  # noqa: E402
    AlarmToken,
    ClosedAlarmPattern,
    MaximumEntropyNextAlarmPredictor,
    accelerated_alarm_alignment,
    charm_closed_alarm_patterns,
    criterion_c_alarm_flood_detection,
    maximum_entropy_single_constraint,
    priority_match_score,
    representative_alarm_patterns,
)


CONFIG_DEFAULT = "configs/experiments/book_chapter5_multidataset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(config_path: Path, data_paths: Sequence[Path]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sources = [
        ROOT / "src/iia_benchmark/models/flood_book.py",
        ROOT / "src/iia_benchmark/data/tep_alarm.py",
        ROOT / "src/iia_benchmark/data/npp_alarm.py",
        ROOT / "src/iia_benchmark/data/fcc.py",
        Path(__file__),
        config_path,
        *data_paths,
    ]
    files = {}
    for path in sources:
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "source_sha256": files,
    }


def trajectory_signature(run: Any) -> str:
    values = np.asarray(run.representation("rising_edge"), dtype=np.int8)
    return hashlib.sha256(values.tobytes()).hexdigest()


def token_sequence(run: Any, *, unique: bool = False) -> tuple[AlarmToken, ...]:
    events = run.to_episode(include_clearances=False).activations()
    output = []
    seen: set[str] = set()
    for event in events:
        if unique and event.tag in seen:
            continue
        seen.add(event.tag)
        # The public payloads do not contain priorities. Encoding every event as
        # p3 avoids fabricating high-priority evidence.
        output.append(AlarmToken(event.tag, float(event.timestamp), 3))
    return tuple(output)


def by_ids(runs: Sequence[Any], ids: Iterable[str]) -> list[Any]:
    wanted = set(ids)
    selected = [run for run in runs if run.run_id in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError("split IDs do not resolve to exactly one run")
    return selected


def grouped_unique_fcc_split(
    runs: Sequence[Any],
    *,
    seed: int,
    train_per_class: int,
    calibration_per_class: int,
    test_per_class: int,
) -> tuple[list[Any], list[Any], list[Any], dict[str, int]]:
    rng = np.random.default_rng(seed)
    partitions = {"train": [], "calibration": [], "test": []}
    duplicate_nonrepresentatives = 0
    unused_unique = 0
    for label in sorted({run.scenario for run in runs}):
        groups: dict[str, list[Any]] = defaultdict(list)
        for run in runs:
            if run.scenario == label:
                groups[trajectory_signature(run)].append(run)
        representatives = []
        for group in groups.values():
            ordered = sorted(group, key=lambda row: row.run_number)
            representatives.append(ordered[int(rng.integers(0, len(ordered)))])
            duplicate_nonrepresentatives += len(ordered) - 1
        representatives = [
            representatives[index] for index in rng.permutation(len(representatives))
        ]
        required = train_per_class + calibration_per_class + test_per_class
        if len(representatives) < required:
            raise RuntimeError(f"FCC class {label} has too few unique trajectories")
        partitions["train"].extend(representatives[:train_per_class])
        partitions["calibration"].extend(
            representatives[train_per_class : train_per_class + calibration_per_class]
        )
        partitions["test"].extend(
            representatives[
                train_per_class
                + calibration_per_class : required
            ]
        )
        unused_unique += len(representatives) - required
    return (
        partitions["train"],
        partitions["calibration"],
        partitions["test"],
        {
            "duplicate_nonrepresentatives": duplicate_nonrepresentatives,
            "unused_unique_trajectories": unused_unique,
        },
    )


def load_bundle(name: str, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    path = ROOT / spec["path"]
    if name == "tep_alarm_dataport":
        runs = list(load_tep_five_class_alarm_runs(path))
        split = build_tep_five_class_split(
            runs,
            train_per_class=spec["train_per_class"],
            calibration_per_class=spec["calibration_per_class"],
            test_per_class=spec["test_per_class"],
            random_state=seed,
            representation="rising_edge",
        )
        train = by_ids(runs, split.train_run_ids)
        calibration = by_ids(runs, split.calibration_run_ids)
        test = by_ids(runs, split.test_run_ids)
        exclusions = {"duplicate_nonrepresentatives": 0, "unused_unique_trajectories": 0}
    elif name == "npp_alarm_dataport":
        runs = list(
            load_npp_alarm_runs(
                path,
                alpha=0.5,
                fault_families=(
                    "FLB", "LLB", "LOCA", "LOCAC", "LR", "RI", "RW",
                    "SGATR", "SGBTR", "SLBIC", "SLBOC",
                ),
                minimum_samples=160,
                horizon_samples=160,
            )
        )
        split = build_npp_alarm_split(
            runs,
            train_per_class=spec["train_per_class"],
            calibration_per_class=spec["calibration_per_class"],
            test_per_class=spec["test_per_class"],
            random_state=seed,
            representation="rising_edge",
        )
        train = by_ids(runs, split.train_run_ids)
        calibration = by_ids(runs, split.calibration_run_ids)
        test = by_ids(runs, split.test_run_ids)
        exclusions = {
            "duplicate_nonrepresentatives": len(split.duplicate_run_ids),
            "cross_label_conflicts": len(split.conflicting_run_ids),
            "unused_unique_trajectories": len(split.unused_run_ids),
        }
    elif name == "fcc_alarm":
        runs = list(load_fcc_alarm_runs(path))
        train, calibration, test, exclusions = grouped_unique_fcc_split(
            runs,
            seed=seed,
            train_per_class=spec["train_per_class"],
            calibration_per_class=spec["calibration_per_class"],
            test_per_class=spec["test_per_class"],
        )
    else:
        raise ValueError(f"unsupported dataset {name}")
    return {
        "name": name,
        "path": path,
        "runs": runs,
        "train": train,
        "calibration": calibration,
        "test": test,
        "match": spec["match"],
        "protocol": spec["protocol"],
        "boundary": spec["boundary"],
        "criterion_c": spec["criterion_c"],
        "exclusions": exclusions,
    }


def prior_gate(bundle: dict[str, Any]) -> dict[str, Any]:
    partitions = [bundle[name] for name in ("train", "calibration", "test")]
    ids = [set(run.run_id for run in part) for part in partitions]
    signatures = [set(trajectory_signature(run) for run in part) for part in partitions]
    all_runs = bundle["runs"]
    signature_labels: dict[str, set[str]] = defaultdict(set)
    for run in all_runs:
        signature_labels[trajectory_signature(run)].add(run.scenario)
    used_signature_labels: dict[str, set[str]] = defaultdict(set)
    for run in (*bundle["train"], *bundle["calibration"], *bundle["test"]):
        used_signature_labels[trajectory_signature(run)].add(run.scenario)
    lengths = np.asarray([len(token_sequence(run)) for run in all_runs], dtype=float)
    unique_lengths = np.asarray(
        [len(token_sequence(run, unique=True)) for run in all_runs], dtype=float
    )
    class_counts = {
        part_name: dict(sorted(Counter(run.scenario for run in bundle[part_name]).items()))
        for part_name in ("train", "calibration", "test")
    }
    checks = {
        "finite_binary_nonempty_matrices": all(
            np.asarray(run.alarm_states).ndim == 2
            and len(run.alarm_states) > 0
            and np.isin(run.alarm_states, [0, 1]).all()
            for run in all_runs
        ),
        "run_id_partitions_disjoint": not (
            ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]
        ),
        "trajectory_partitions_disjoint": not (
            signatures[0] & signatures[1]
            or signatures[0] & signatures[2]
            or signatures[1] & signatures[2]
        ),
        "no_cross_label_identical_trajectory_in_used_split": not any(
            len(labels) > 1 for labels in used_signature_labels.values()
        ),
        "all_test_classes_seen_in_train": set(class_counts["test"]) <= set(class_counts["train"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_runs": len(all_runs),
        "partition_runs": {name: len(bundle[name]) for name in ("train", "calibration", "test")},
        "class_counts": class_counts,
        "trajectory_signatures": len(signature_labels),
        "cross_label_conflicting_signatures": sum(
            len(labels) > 1 for labels in signature_labels.values()
        ),
        "used_cross_label_conflicting_signatures": sum(
            len(labels) > 1 for labels in used_signature_labels.values()
        ),
        "activation_sequence_length": {
            "minimum": int(np.min(lengths)),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "maximum": int(np.max(lengths)),
            "empty_runs": int(np.sum(lengths == 0)),
        },
        "unique_first_activation_length": {
            "minimum": int(np.min(unique_lengths)),
            "median": float(np.median(unique_lengths)),
            "p95": float(np.quantile(unique_lengths, 0.95)),
            "maximum": int(np.max(unique_lengths)),
        },
        "exclusions": bundle["exclusions"],
    }


def jaccard(left: set[str], right: set[str] | frozenset[str]) -> float:
    union = left | set(right)
    return len(left & set(right)) / len(union) if union else 1.0


def evaluate_criterion_c(bundle: dict[str, Any]) -> dict[str, Any]:
    parameters = {key: int(value) for key, value in bundle["criterion_c"].items()}
    records = []
    for run in bundle["test"]:
        result = criterion_c_alarm_flood_detection(
            run.alarm_states,
            tag_names=run.alarm_names,
            **parameters,
        )
        starts = int(
            np.sum(
                (result.delayed_detection == 1)
                & (np.r_[0, result.delayed_detection[:-1]] == 0)
            )
        )
        records.append(
            {
                "run_id": run.run_id,
                "label": run.scenario,
                "maximum_cardinality": int(np.max(result.cardinality, initial=0)),
                "candidate_intervals": starts,
                "candidate_exposure": float(np.mean(result.delayed_detection)) if len(result.delayed_detection) else 0.0,
            }
        )
    intervals = sum(row["candidate_intervals"] for row in records)
    maximum = max((row["maximum_cardinality"] for row in records), default=0)
    return {
        "parameters": parameters,
        "evaluated_runs": len(records),
        "candidate_intervals": intervals,
        "runs_with_candidates": sum(row["candidate_intervals"] > 0 for row in records),
        "candidate_episode_rate": float(np.mean([row["candidate_intervals"] > 0 for row in records])),
        "mean_candidate_exposure": float(np.mean([row["candidate_exposure"] for row in records])),
        "maximum_attention_cardinality": maximum,
        "mechanism_activation_passed": intervals > 0 and maximum >= parameters["threshold"],
        "performance_activation_passed": None,
        "competitive_credit_passed": None,
        "activation_passed": intervals > 0 and maximum >= parameters["threshold"],
        "supervised_interval_score_reportable": False,
        "score_blocker": "no expert-confirmed alarm-flood start/end labels",
    }


def evaluate_alignment(bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(config)
    prototypes_per_class = int(parameters.pop("prototypes_per_class"))
    parameters["time_tolerance"] = float(parameters.pop("time_tolerance_seconds"))
    parameters.pop("priority_boundary", None)
    classes = sorted({run.scenario for run in bundle["train"]})
    prototypes = {
        label: [
            token_sequence(run)
            for run in bundle["train"]
            if run.scenario == label
        ][:prototypes_per_class]
        for label in classes
    }
    truth, prediction, baseline_prediction, best_scores = [], [], [], []
    cells = 0
    for run in bundle["test"]:
        query = token_sequence(run)
        query_set = {token.tag for token in query}
        scores, baseline_scores = {}, {}
        for label, candidates in prototypes.items():
            rows = [accelerated_alarm_alignment(query, candidate, **parameters) for candidate in candidates]
            cells += sum(row.cells_evaluated for row in rows)
            scores[label] = max((row.similarity for row in rows), default=0.0)
            baseline_scores[label] = max(
                (jaccard(query_set, {token.tag for token in candidate}) for candidate in candidates),
                default=0.0,
            )
        truth.append(run.scenario)
        prediction.append(max(scores, key=scores.get))
        baseline_prediction.append(max(baseline_scores, key=baseline_scores.get))
        best_scores.append(max(scores.values()))
    metrics = multiclass_classification_metrics(truth, prediction)
    baseline = multiclass_classification_metrics(truth, baseline_prediction)
    chance = 1.0 / len(classes)
    mechanism = len(set(prediction)) >= 2 and max(best_scores, default=0.0) > 0
    performance = metrics["balanced_accuracy"] >= max(chance * 1.5, chance + 0.05)
    competitive = metrics["balanced_accuracy"] > baseline["balanced_accuracy"]
    return {
        "metrics": metrics,
        "set_jaccard_baseline": baseline,
        "prototype_runs_per_class": prototypes_per_class,
        "parameters": parameters,
        "mean_best_similarity": float(np.mean(best_scores)),
        "cells_evaluated": cells,
        "predicted_classes": len(set(prediction)),
        "chance_balanced_accuracy": chance,
        "mechanism_activation_passed": mechanism,
        "performance_activation_passed": performance,
        "competitive_credit_passed": competitive,
        "activation_passed": mechanism and performance,
        "priority_weighting_real_data_tested": False,
    }


def evaluate_patterns(bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    classes = sorted({run.scenario for run in bundle["train"]})
    models = {}
    for label in classes:
        transactions = [
            set(token.tag for token in token_sequence(run, unique=True))
            for run in bundle["train"]
            if run.scenario == label
        ]
        closed = charm_closed_alarm_patterns(
            transactions, minimum_support=float(config["minimum_support"])
        )
        representatives = representative_alarm_patterns(
            closed, similarity_threshold=float(config["similarity_threshold"])
        )
        models[label] = {"closed": closed, "representatives": representatives}
    missing = [label for label, model in models.items() if not model["representatives"]]
    if missing:
        return {
            "activation_passed": False,
            "mechanism_activation_passed": False,
            "performance_activation_passed": False,
            "competitive_credit_passed": False,
            "missing_representative_classes": missing,
            "paper_score_closure": False,
        }
    core_patterns = {}
    for label in classes:
        transactions = [
            set(token.tag for token in token_sequence(run, unique=True))
            for run in bundle["train"]
            if run.scenario == label
        ]
        counts = Counter(tag for transaction in transactions for tag in transaction)
        core_patterns[label] = {
            tag for tag, count in counts.items() if count >= np.ceil(0.5 * len(transactions))
        }
    truth, prediction, baseline_prediction, scores = [], [], [], []
    for run in bundle["test"]:
        transaction = set(token.tag for token in token_sequence(run, unique=True))
        class_scores = {
            label: max(
                jaccard(transaction, representative.items)
                for representative in model["representatives"]
            )
            for label, model in models.items()
        }
        truth.append(run.scenario)
        prediction.append(max(class_scores, key=class_scores.get))
        baseline_prediction.append(
            max(core_patterns, key=lambda label: jaccard(transaction, core_patterns[label]))
        )
        scores.append(max(class_scores.values()))
    counts = {
        label: {
            "closed_patterns": len(model["closed"]),
            "representative_patterns": len(model["representatives"]),
        }
        for label, model in models.items()
    }
    metrics = multiclass_classification_metrics(truth, prediction)
    baseline = multiclass_classification_metrics(truth, baseline_prediction)
    chance = 1.0 / len(classes)
    mechanism = len(set(prediction)) >= 2
    performance = metrics["balanced_accuracy"] >= max(chance * 1.5, chance + 0.05)
    competitive = metrics["balanced_accuracy"] > baseline["balanced_accuracy"]
    return {
        "metrics": metrics,
        "class_core_jaccard_baseline": baseline,
        "parameters": {
            "minimum_support": config["minimum_support"],
            "similarity_threshold": config["similarity_threshold"],
        },
        "pattern_counts": counts,
        "total_closed_patterns": sum(row["closed_patterns"] for row in counts.values()),
        "total_representative_patterns": sum(row["representative_patterns"] for row in counts.values()),
        "compression_ratio": sum(row["representative_patterns"] for row in counts.values()) / sum(row["closed_patterns"] for row in counts.values()),
        "mean_best_jaccard": float(np.mean(scores)),
        "predicted_classes": len(set(prediction)),
        "chance_balanced_accuracy": chance,
        "mechanism_activation_passed": mechanism,
        "performance_activation_passed": performance,
        "competitive_credit_passed": competitive,
        "activation_passed": mechanism and performance,
    }


def evaluate_maximum_entropy(bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    fit_sequences = [
        token_sequence(run, unique=True)
        for run in bundle["train"]
        if len(token_sequence(run, unique=True)) >= 2
    ]
    predictor = MaximumEntropyNextAlarmPredictor(
        time_scale=float(config["time_scale_seconds"]),
        regularization=float(config["regularization"]),
        max_iterations=int(config["max_iterations"]),
    ).fit(fit_sequences)
    target_counts = Counter(
        sequence[index].tag
        for sequence in fit_sequences
        for index in range(1, len(sequence))
    )
    truth, prediction, baseline_prediction = [], [], []
    top3_hits = 0
    losses, brier, lead_times = [], [], []
    candidate_transitions = 0
    nonuniform = 0
    vocabulary = tuple(predictor.vocabulary_)
    for run in bundle["test"]:
        sequence = token_sequence(run, unique=True)
        for index in range(1, len(sequence)):
            candidate_transitions += 1
            target = sequence[index].tag
            probabilities = predictor.predict_proba(sequence[:index])
            if target not in probabilities or not probabilities:
                continue
            ranking = sorted(probabilities, key=probabilities.get, reverse=True)
            candidates = [tag for tag in target_counts if tag not in {token.tag for token in sequence[:index]}]
            if not candidates:
                continue
            truth.append(target)
            prediction.append(ranking[0])
            baseline_prediction.append(max(candidates, key=target_counts.get))
            top3_hits += int(target in ranking[:3])
            probability = max(float(probabilities[target]), 1e-15)
            losses.append(-np.log(probability))
            vector = np.asarray([probabilities.get(tag, 0.0) for tag in vocabulary])
            target_vector = np.asarray([tag == target for tag in vocabulary], dtype=float)
            brier.append(float(np.mean((vector - target_vector) ** 2)))
            lead_times.append(sequence[index].timestamp - sequence[index - 1].timestamp)
            nonuniform += int(np.ptp(np.asarray(list(probabilities.values()))) > 1e-12)
    if not truth:
        return {
            "activation_passed": False,
            "mechanism_activation_passed": False,
            "performance_activation_passed": False,
            "competitive_credit_passed": False,
            "evaluated_transitions": 0,
        }
    metrics = multiclass_classification_metrics(truth, prediction)
    baseline = multiclass_classification_metrics(truth, baseline_prediction)
    mechanism = nonuniform > 0 and len(set(prediction)) >= 2
    performance = metrics["macro_f1"] >= 0.8
    competitive = (
        metrics["accuracy"] > baseline["accuracy"]
        and metrics["macro_f1"] > baseline["macro_f1"]
    )
    return {
        "metrics": {
            "top1_accuracy": metrics["accuracy"],
            "top3_accuracy": top3_hits / len(truth),
            "macro_f1_eta_surrogate": metrics["macro_f1"],
            "negative_log_likelihood": float(np.mean(losses)),
            "brier_score": float(np.mean(brier)),
            "mean_lead_time_seconds": float(np.mean(lead_times)),
            "vocabulary_coverage": len(truth) / candidate_transitions,
        },
        "global_frequency_baseline": {
            "top1_accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
        },
        "learned_weights": predictor.weights_.tolist(),
        "vocabulary_size": len(vocabulary),
        "candidate_transitions": candidate_transitions,
        "evaluated_transitions": len(truth),
        "predicted_tags": len(set(prediction)),
        "nonuniform_predictions": nonuniform,
        "mechanism_activation_passed": mechanism,
        "performance_activation_passed": performance,
        "competitive_credit_passed": competitive,
        "activation_passed": mechanism and performance,
        "eta_boundary": "macro F1 is a transfer surrogate, not Eq. 5.85 under the paper's 10-fold Monte Carlo protocol",
    }


def named_book_items() -> dict[str, Any]:
    scores = [priority_match_score(level, 3) for level in (1, 2, 3)]
    first = [AlarmToken(str(tag), index, 3) for index, tag in enumerate((3, 2, 1, 4, 3, 2, 2))]
    second = [AlarmToken(str(tag), index, 3) for index, tag in enumerate((3, 4, 2, 1, 4, 2))]
    alignment = accelerated_alarm_alignment(
        first, second, seed_length=3, max_seeds=7, extension_band=10
    )
    pair_tags = [(first[i].tag, second[j].tag) for i, j in alignment.aligned_pairs]
    patterns = tuple(
        ClosedAlarmPattern(frozenset(map(str, items)), frozenset({index}), 0.2)
        for index, items in enumerate(
            ({2, 3, 4, 5}, {1, 3, 4, 5}, {1, 2, 4, 5}, {1, 2, 3, 5}, {1, 2, 3, 4})
        )
    )
    representatives = representative_alarm_patterns(patterns, similarity_threshold=1 / 3)
    candidates = ("x3", "x4", "x5")
    constraints = [
        maximum_entropy_single_constraint(candidates, constrained_candidate=tag, constrained_probability=probability)
        for tag, probability in (("x3", 3 / 20), ("x4", 4 / 5), ("x5", 1 / 20))
    ]
    criterion_states = np.zeros((30, 4), dtype=np.int8)
    criterion_states[2:, 0] = 1
    criterion_states[12:18, 1] = 1
    criterion_states[20:, 2] = 1
    criterion = criterion_c_alarm_flood_detection(
        criterion_states,
        tag_names=("standing", "recent", "new", "idle"),
        attention_window=10,
        long_standing_window=20,
        update_step=10,
        threshold=1,
    )
    return {
        "criterion_c_equations_5_5_to_5_7": {
            "passed": "standing" in criterion.attention_sets[0]
            and "standing" not in criterion.attention_sets[-1]
            and "new" in criterion.attention_sets[-1],
            "scope": "inheritance and long-standing exclusion invariant",
        },
        "table_5_5_priority_scores": {
            "observed": scores,
            "expected": [6.0, 4.5, 3.0],
            "passed": np.allclose(scores, [6.0, 4.5, 3.0]),
        },
        "equation_5_16_alignment": {
            "observed_matched_tags": pair_tags,
            "expected_match_count": 5,
            "passed": len(pair_tags) == 5 and all(left == right for left, right in pair_tags),
        },
        "section_5_3_five_pattern_compression": {
            "input_patterns": 5,
            "representative_patterns": len(representatives),
            "representative_union": sorted(representatives[0].items),
            "passed": len(representatives) == 1
            and representatives[0].items == frozenset({"1", "2", "3", "4", "5"}),
        },
        "table_5_15_maximum_entropy": {
            "observed_multipliers": [row.lagrange_multiplier for row in constraints],
            "expected_multipliers": [-1.0414, 2.0794, -2.2513],
            "observed_x4_probability": constraints[1].probabilities["x4"],
            "book_x4_probability": 0.7999,
            "passed": np.allclose(
                [row.lagrange_multiplier for row in constraints],
                [-1.0414, 2.0794, -2.2513],
                atol=5e-5,
            )
            and abs(constraints[1].probabilities["x4"] - 0.7999) <= 2e-4,
        },
    }


def run(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result = {
        "seed": seed,
        "named_book_items": named_book_items(),
        "datasets": {},
        "paper_score_closure": {
            algorithm: False for algorithm in config["algorithms"]
        },
        "original_data_blockers": {
            "book_5_1_flood_detection": "the 2,226-variable thermal-power-plant alarm history and expert flood intervals are unavailable",
            "book_5_2_alarm_alignment": "the 389-flood oil-conversion database and Table 5.12 query sequences are unavailable",
            "book_5_3_closed_patterns": "the April industrial transaction database producing 921 closed and 207 representative patterns is unavailable",
            "book_5_4_max_entropy_prediction": "the 26 TEP historical flood sequences and Monte Carlo payload for Tables 5.18-5.22 and Figures 5.30-5.31 are unavailable",
        },
        "reporting_boundary": config["reporting_boundary"],
    }
    for name, spec in config["datasets"].items():
        started = time.perf_counter()
        bundle = load_bundle(name, spec, seed)
        prior = prior_gate(bundle)
        if not prior["passed"]:
            raise RuntimeError(f"{name} failed G0 prior gate: {prior['checks']}")
        print(f"[{name}] G0 passed; running four Chapter 5 methods", flush=True)
        method_times: dict[str, float] = {}
        method_started = time.perf_counter()
        criterion_result = evaluate_criterion_c(bundle)
        method_times["book_5_1_flood_detection"] = time.perf_counter() - method_started
        print(f"[{name}] Criterion C completed", flush=True)
        method_started = time.perf_counter()
        alignment_result = evaluate_alignment(bundle, config["alignment"])
        method_times["book_5_2_alarm_alignment"] = time.perf_counter() - method_started
        print(f"[{name}] alignment completed", flush=True)
        method_started = time.perf_counter()
        pattern_result = evaluate_patterns(bundle, config["closed_patterns"])
        method_times["book_5_3_closed_patterns"] = time.perf_counter() - method_started
        print(f"[{name}] CHARM completed", flush=True)
        method_started = time.perf_counter()
        entropy_result = evaluate_maximum_entropy(bundle, config["maximum_entropy"])
        method_times["book_5_4_max_entropy_prediction"] = time.perf_counter() - method_started
        print(f"[{name}] maximum entropy completed", flush=True)
        result["datasets"][name] = {
            "match": bundle["match"],
            "protocol": bundle["protocol"],
            "boundary": bundle["boundary"],
            "prior_gate": prior,
            "book_5_1_flood_detection": criterion_result,
            "book_5_2_alarm_alignment": alignment_result,
            "book_5_3_closed_patterns": pattern_result,
            "book_5_4_max_entropy_prediction": entropy_result,
            "method_wall_clock_seconds": method_times,
            "wall_clock_seconds": time.perf_counter() - started,
        }
        print(
            f"[{name}] completed in {result['datasets'][name]['wall_clock_seconds']:.1f}s",
            flush=True,
        )
    result["mechanism_activation"] = {
        algorithm: {
            dataset: bool(rows[algorithm]["mechanism_activation_passed"])
            for dataset, rows in result["datasets"].items()
        }
        for algorithm in config["algorithms"]
    }
    result["performance_activation"] = {
        algorithm: {
            dataset: rows[algorithm]["performance_activation_passed"]
            for dataset, rows in result["datasets"].items()
        }
        for algorithm in config["algorithms"]
    }
    result["competitive_credit"] = {
        algorithm: {
            dataset: rows[algorithm]["competitive_credit_passed"]
            for dataset, rows in result["datasets"].items()
        }
        for algorithm in config["algorithms"]
    }
    result["engineering_activation"] = {
        algorithm: {
            dataset: bool(rows[algorithm]["activation_passed"])
            for dataset, rows in result["datasets"].items()
        }
        for algorithm in config["algorithms"]
    }
    result["named_reproduction_passed"] = all(
        row["passed"] for row in result["named_book_items"].values()
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--config", default=CONFIG_DEFAULT, type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    run_name = out_dir.name
    run_index = int(run_name.split("_")[-1]) - 1
    seed = int(config["seeds"][run_index])
    data_paths = [ROOT / spec["path"] for spec in config["datasets"].values()]
    started = datetime.now(timezone.utc)
    provenance = source_provenance(config_path, data_paths)
    before = time.perf_counter()
    payload = run(config, seed)
    duration = time.perf_counter() - before
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "schema_version": 1,
        "run_name": run_name,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "git_revision": revision,
        "execution_provenance": provenance,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": duration,
        "result": payload,
    }
    (out_dir / "final_info.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_name": run_name,
        "seed": seed,
        "wall_clock_seconds": duration,
        "named_reproduction_passed": payload["named_reproduction_passed"],
        "engineering_activation": payload["engineering_activation"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
