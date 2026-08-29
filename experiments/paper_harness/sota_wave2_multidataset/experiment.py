#!/usr/bin/env python3
"""Run SOTA Wave 2 on grouped TEP, NPP, and FCC alarm trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data.fcc import load_fcc_alarm_runs  # noqa: E402
from iia_benchmark.data.npp_alarm import (  # noqa: E402
    build_npp_alarm_split,
    load_npp_alarm_runs,
)
from iia_benchmark.data.schema import AlarmEpisode, AlarmEvent  # noqa: E402
from iia_benchmark.data.tep_alarm import (  # noqa: E402
    build_tep_five_class_split,
    load_tep_five_class_alarm_runs,
)
from iia_benchmark.evaluation import (  # noqa: E402
    PerturbationScenario,
    multiclass_classification_metrics,
    run_afc_robustness_benchmark,
)
from iia_benchmark.models import (  # noqa: E402
    BifurcationDelayTimer,
    BifurcationForecast,
    CASIMClassifier,
    ConEAlarmFloodClassifier,
    CrossConformalAlarmFloodClassifier,
    CTFHAlarmFloodClassifier,
    HDAMTemplateMatcher,
    ModifiedTFIDFVectorizer,
    OptimalTimeEncodedHistogramClassifier,
    SpectralAlarmFloodClusterer,
    TFIDFLSTMAlarmFloodClassifier,
    TimedAlarmSequence,
    UncertaintyReductionForecaster,
    extract_bifurcation_training_data,
    optimize_ngram_size,
)


CONFIG_DEFAULT = "configs/experiments/sota_wave2_multidataset.json"
POINT_MODEL_IDS = (
    "jaccard_class_core",
    "ctfh_fingerprinting",
    "structured_hdam",
    "casim",
    "modified_tfidf_afc",
    "time_encoded_histogram_hybrid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(config_path: Path, data_paths: Sequence[Path]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sources = (
        ROOT / "src/iia_benchmark/data/tep_alarm.py",
        ROOT / "src/iia_benchmark/data/npp_alarm.py",
        ROOT / "src/iia_benchmark/data/fcc.py",
        ROOT / "src/iia_benchmark/evaluation/robustness.py",
        ROOT / "src/iia_benchmark/models/ctfh.py",
        ROOT / "src/iia_benchmark/models/structured_hdam.py",
        ROOT / "src/iia_benchmark/models/casim.py",
        ROOT / "src/iia_benchmark/models/modified_tfidf_afc.py",
        ROOT / "src/iia_benchmark/models/time_encoded_histogram.py",
        ROOT / "src/iia_benchmark/models/cone_afc.py",
        ROOT / "src/iia_benchmark/models/cross_conformal_afc.py",
        ROOT / "src/iia_benchmark/models/uncertainty_reduction.py",
        Path(__file__),
        config_path,
        *data_paths,
    )
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in sources
            if path.is_file()
        },
    }


def trajectory_signature(run: Any) -> str:
    values = np.asarray(run.representation("rising_edge"), dtype=np.int8)
    return hashlib.sha256(values.tobytes()).hexdigest()


def by_ids(runs: Sequence[Any], identifiers: Iterable[str]) -> list[Any]:
    lookup = {run.run_id: run for run in runs}
    ordered = list(identifiers)
    if len(lookup) != len(runs) or any(identifier not in lookup for identifier in ordered):
        raise RuntimeError("split IDs do not resolve to unique runs")
    return [lookup[identifier] for identifier in ordered]


def grouped_unique_fcc_split(
    runs: Sequence[Any],
    *,
    seed: int,
    train_per_class: int,
    calibration_per_class: int,
    test_per_class: int,
) -> tuple[list[Any], list[Any], list[Any], dict[str, int]]:
    rng = np.random.default_rng(seed)
    partitions: dict[str, list[Any]] = {"train": [], "calibration": [], "test": []}
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
            representatives[train_per_class + calibration_per_class : required]
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


def alarm_tensor(runs: Sequence[Any]) -> np.ndarray:
    return np.stack(
        [np.asarray(run.representation("rising_edge"), dtype=np.float32).T for run in runs]
    )


def activation_episode(run: Any) -> AlarmEpisode:
    rising = np.asarray(run.representation("rising_edge"), dtype=np.int8)
    if hasattr(run, "sample_seconds"):
        sample_seconds = float(run.sample_seconds)
    else:
        sample_seconds = float(run.sample_minutes) * 60.0
    events = tuple(
        AlarmEvent(float(sample) * sample_seconds, run.alarm_names[int(column)], 1, 1)
        for sample, column in np.argwhere(rising > 0)
    )
    return AlarmEpisode(
        episode_id=run.run_id,
        events=events,
        label=str(run.scenario),
        root_cause=str(run.scenario),
    )


def load_bundle(name: str, specification: dict[str, Any], seed: int) -> dict[str, Any]:
    path = ROOT / specification["path"]
    if name == "tep_alarm_dataport":
        runs = list(load_tep_five_class_alarm_runs(path))
        split = build_tep_five_class_split(
            runs,
            train_per_class=int(specification["train_per_class"]),
            calibration_per_class=int(specification["calibration_per_class"]),
            test_per_class=int(specification["test_per_class"]),
            random_state=seed,
            representation="rising_edge",
        )
        train = by_ids(runs, split.train_run_ids)
        calibration = by_ids(runs, split.calibration_run_ids)
        test = by_ids(runs, split.test_run_ids)
        exclusions = {"duplicate_nonrepresentatives": 0, "cross_label_conflicts": 0}
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
            train_per_class=int(specification["train_per_class"]),
            calibration_per_class=int(specification["calibration_per_class"]),
            test_per_class=int(specification["test_per_class"]),
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
            train_per_class=int(specification["train_per_class"]),
            calibration_per_class=int(specification["calibration_per_class"]),
            test_per_class=int(specification["test_per_class"]),
        )
    else:
        raise ValueError(f"unsupported dataset: {name}")
    sample_seconds = (
        float(runs[0].sample_seconds)
        if hasattr(runs[0], "sample_seconds")
        else float(runs[0].sample_minutes) * 60.0
    )
    names = tuple(runs[0].alarm_names)
    length = int(runs[0].alarm_states.shape[0])
    partitions = {"train": train, "calibration": calibration, "test": test}
    return {
        "name": name,
        "path": path,
        "all_runs": runs,
        "partitions": partitions,
        "X": {key: alarm_tensor(value) for key, value in partitions.items()},
        "y": {
            key: np.asarray([str(run.scenario) for run in value], dtype=object)
            for key, value in partitions.items()
        },
        "episodes": {
            key: tuple(activation_episode(run) for run in value)
            for key, value in partitions.items()
        },
        "alarm_names": names,
        "sample_seconds": sample_seconds,
        "length": length,
        "match": specification["match"],
        "protocol": specification["protocol"],
        "boundary": specification["boundary"],
        "robustness_test_per_class": int(specification["robustness_test_per_class"]),
        "exclusions": exclusions,
    }


def episode_tensor(episodes: Sequence[AlarmEpisode], bundle: dict[str, Any]) -> np.ndarray:
    names = bundle["alarm_names"]
    name_to_column = {name: index for index, name in enumerate(names)}
    values = np.zeros(
        (len(episodes), len(names), bundle["length"]), dtype=np.float32
    )
    for row, episode in enumerate(episodes):
        for event in episode.activations():
            column = name_to_column.get(event.tag)
            sample = int(np.rint(float(event.timestamp) / bundle["sample_seconds"]))
            if column is not None and 0 <= sample < bundle["length"]:
                values[row, column, sample] = 1.0
    return values


def episode_tokens(episode: AlarmEpisode) -> tuple[str, ...]:
    return tuple(event.tag for event in sorted(episode.activations(), key=lambda row: row.timestamp))


def timed_sequence(episode: AlarmEpisode) -> TimedAlarmSequence | None:
    events = sorted(episode.activations(), key=lambda row: row.timestamp)
    if not events:
        return None
    return TimedAlarmSequence(
        np.asarray([event.timestamp for event in events], dtype=float),
        tuple(event.tag for event in events),
    )


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def prior_gate(bundle: dict[str, Any]) -> dict[str, Any]:
    partitions = bundle["partitions"]
    id_sets = {key: {run.run_id for run in value} for key, value in partitions.items()}
    signatures = {
        key: {trajectory_signature(run) for run in value}
        for key, value in partitions.items()
    }
    clean_raster_equal = all(
        np.array_equal(
            episode_tensor((activation_episode(run),), bundle)[0],
            np.asarray(run.representation("rising_edge"), dtype=np.float32).T,
        )
        for run in (*partitions["train"], *partitions["calibration"], *partitions["test"])
    )
    used = (*partitions["train"], *partitions["calibration"], *partitions["test"])
    event_counts = [int(np.sum(run.representation("rising_edge"))) for run in used]
    unique_tags = [
        int(np.sum(np.any(run.representation("rising_edge") > 0, axis=0))) for run in used
    ]
    all_labels = sorted({str(run.scenario) for run in used})
    label_tag_sets = {
        label: [
            frozenset(activation_episode(run).tags(unique=True))
            for run in used
            if str(run.scenario) == label
        ]
        for label in all_labels
    }
    within, across = [], []
    for left_index, left_label in enumerate(all_labels):
        for left_pos, left in enumerate(label_tag_sets[left_label]):
            for right_label in all_labels[left_index:]:
                start = left_pos + 1 if right_label == left_label else 0
                for right in label_tag_sets[right_label][start:]:
                    union = left | right
                    similarity = len(left & right) / len(union) if union else 1.0
                    (within if left_label == right_label else across).append(similarity)
    checks = {
        "finite_binary_nonempty": all(
            value.ndim == 3
            and value.shape[0] > 0
            and np.isfinite(value).all()
            and np.isin(value, [0, 1]).all()
            for value in bundle["X"].values()
        ),
        "disjoint_run_ids": not (
            id_sets["train"] & id_sets["calibration"]
            or id_sets["train"] & id_sets["test"]
            or id_sets["calibration"] & id_sets["test"]
        ),
        "disjoint_complete_trajectory_hashes": not (
            signatures["train"] & signatures["calibration"]
            or signatures["train"] & signatures["test"]
            or signatures["calibration"] & signatures["test"]
        ),
        "clean_event_raster_equals_registered_rising_edge": clean_raster_equal,
        "class_coverage_all_partitions": all(
            set(bundle["y"][key].tolist()) == set(all_labels)
            for key in ("train", "calibration", "test")
        ),
        "nonempty_activation_sequences": min(event_counts) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "class_counts": {
            key: dict(sorted(Counter(bundle["y"][key].tolist()).items()))
            for key in ("train", "calibration", "test")
        },
        "shape": {
            key: list(bundle["X"][key].shape) for key in ("train", "calibration", "test")
        },
        "event_count_distribution": percentile_summary(event_counts),
        "unique_tag_distribution": percentile_summary(unique_tags),
        "mean_within_class_tag_jaccard": float(np.mean(within)),
        "mean_cross_class_tag_jaccard": float(np.mean(across)),
        "trajectory_hash_counts": {key: len(value) for key, value in signatures.items()},
        "exclusions": bundle["exclusions"],
    }


def paired_gate(candidate: Sequence[float], parent: Sequence[float], threshold: float) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(parent, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("paired gate inputs must be equal nonempty vectors")
    delta = left - right
    mean_delta = float(np.mean(delta))
    standard_deviation = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
    if standard_deviation == 0.0:
        z_score = None
        passed = mean_delta > 0.0
        decision = "deterministic_positive" if passed else "no_positive_delta"
    else:
        z_score = float(mean_delta / (standard_deviation / np.sqrt(delta.size)))
        passed = mean_delta > 0.0 and z_score >= threshold
        decision = "paired_z"
    return {
        "mean_delta": mean_delta,
        "standard_deviation": standard_deviation,
        "z_score": z_score,
        "pairs": int(delta.size),
        "threshold": threshold,
        "passed": bool(passed),
        "decision": decision,
    }


class JaccardClassCore:
    def fit(self, episodes: Sequence[AlarmEpisode], labels: np.ndarray) -> "JaccardClassCore":
        self.classes_ = np.asarray(sorted(set(labels.tolist())))
        profiles = []
        for label in self.classes_:
            samples = [
                frozenset(episode_tokens(episode))
                for episode, item in zip(episodes, labels)
                if item == label
            ]
            counts = Counter(tag for sample in samples for tag in sample)
            threshold = 0.5 * len(samples)
            profiles.append(frozenset(tag for tag, count in counts.items() if count >= threshold))
        self.profiles_ = tuple(profiles)
        return self

    def predict(self, episodes: Sequence[AlarmEpisode]) -> np.ndarray:
        output = []
        for episode in episodes:
            query = frozenset(episode_tokens(episode))
            scores = []
            for profile in self.profiles_:
                union = query | profile
                scores.append(len(query & profile) / len(union) if union else 1.0)
            output.append(self.classes_[int(np.argmax(scores))])
        return np.asarray(output, dtype=object)


def predict_point_model(
    model_id: str,
    model: Any,
    episodes: Sequence[AlarmEpisode],
    bundle: dict[str, Any],
) -> np.ndarray:
    if model_id == "jaccard_class_core":
        return model.predict(episodes)
    majority = str(Counter(bundle["y"]["train"].tolist()).most_common(1)[0][0])
    if model_id in {"ctfh_fingerprinting", "structured_hdam", "casim"}:
        values = episode_tensor(episodes, bundle)
        probabilities = np.asarray(model.predict_proba(values), dtype=float)
        return np.asarray(model.classes_, dtype=object)[np.argmax(probabilities, axis=1)]
    nonempty = [index for index, episode in enumerate(episodes) if episode_tokens(episode)]
    output = np.full(len(episodes), majority, dtype=object)
    if not nonempty:
        return output
    if model_id == "modified_tfidf_afc":
        sequences = [episode_tokens(episodes[index]) for index in nonempty]
        output[nonempty] = model.predict(sequences)
    elif model_id == "time_encoded_histogram_hybrid":
        sequences = [timed_sequence(episodes[index]) for index in nonempty]
        if any(sequence is None for sequence in sequences):
            raise RuntimeError("nonempty event sequences failed timed conversion")
        output[nonempty] = model.predict(sequences)
    else:
        raise ValueError(f"unsupported point model: {model_id}")
    return output


def fit_point_models(
    bundle: dict[str, Any], configuration: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    parameters = configuration["classification"]
    X_train = bundle["X"]["train"]
    y_train = bundle["y"]["train"]
    train_episodes = bundle["episodes"]["train"]
    models: dict[str, Any] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    started = time.perf_counter()
    baseline = JaccardClassCore().fit(train_episodes, y_train)
    models["jaccard_class_core"] = baseline
    diagnostics["jaccard_class_core"] = {
        "fit_seconds": time.perf_counter() - started,
        "profile_sizes": [len(profile) for profile in baseline.profiles_],
        "mechanism_pass": any(baseline.profiles_),
    }

    print("CTFH tests whether deterministic consensus fingerprints activate on each real alarm payload.")
    started = time.perf_counter()
    ctfh = CTFHAlarmFloodClassifier(**parameters["ctfh_fingerprinting"]).fit(
        X_train, y_train
    )
    models["ctfh_fingerprinting"] = ctfh
    diagnostics["ctfh_fingerprinting"] = {
        "fit_seconds": time.perf_counter() - started,
        "consensus_hashes": [len(profile.hashes) for profile in ctfh.profiles_],
        "variability": [profile.variability_index for profile in ctfh.profiles_],
        "mechanism_pass": sum(len(profile.hashes) for profile in ctfh.profiles_) > 0,
    }

    print("HDAM tests whether category templates remain finite and discriminative under a prefix-compatible width.")
    started = time.perf_counter()
    hdam_parameters = dict(parameters["structured_hdam"])
    width_fraction = float(hdam_parameters.pop("template_width_fraction"))
    binned_length = int(np.ceil(bundle["length"] / int(hdam_parameters["bin_size"])))
    template_width = max(1, int(round(binned_length * width_fraction)))
    hdam = HDAMTemplateMatcher(template_width=template_width, **hdam_parameters).fit(
        X_train, y_train
    )
    models["structured_hdam"] = hdam
    diagnostics["structured_hdam"] = {
        "fit_seconds": time.perf_counter() - started,
        "template_width": template_width,
        "template_stability": [template.stability for template in hdam.templates_],
        "mechanism_pass": bool(
            hdam.templates_ and np.isfinite([template.stability for template in hdam.templates_]).all()
        ),
    }

    print("CASIM tests the MultiRocket-ridge ensemble on the same leakage-controlled tensor split.")
    started = time.perf_counter()
    casim_parameters = dict(parameters["casim"])
    casim_parameters["alphas"] = tuple(casim_parameters["alphas"])
    casim = CASIMClassifier(random_state=seed, **casim_parameters).fit(X_train, y_train)
    models["casim"] = casim
    diagnostics["casim"] = {
        "fit_seconds": time.perf_counter() - started,
        "ensemble_classifiers": len(casim.classifiers_),
        "features_per_classifier": casim.n_features,
        "loop_training_samples": len(casim.loop_training_labels_),
        "mechanism_pass": len(casim.classifiers_) == int(casim_parameters["n_classifiers"]),
    }

    print("Modified TF-IDF tests the published 1-4 gram selection item before LSTM early classification.")
    started = time.perf_counter()
    tfidf_parameters = dict(parameters["modified_tfidf_afc"])
    candidates = tuple(tfidf_parameters.pop("ngram_candidates"))
    train_sequences = [episode_tokens(episode) for episode in train_episodes]
    selection = optimize_ngram_size(
        train_sequences,
        candidates,
        len(set(y_train.tolist())),
        position_decay=float(tfidf_parameters["position_decay"]),
        random_state=seed,
    )
    selected_features = ModifiedTFIDFVectorizer(
        selection.ngram_size, float(tfidf_parameters["position_decay"])
    ).fit_transform(train_sequences)
    cluster_labels = SpectralAlarmFloodClusterer(
        len(set(y_train.tolist())), random_state=seed
    ).fit_predict(selected_features)
    from sklearn.metrics import adjusted_rand_score

    clustering_ari = float(adjusted_rand_score(y_train, cluster_labels))
    tfidf = TFIDFLSTMAlarmFloodClassifier(
        ngram_size=selection.ngram_size,
        random_state=seed,
        **tfidf_parameters,
    ).fit(train_sequences, y_train)
    models["modified_tfidf_afc"] = tfidf
    diagnostics["modified_tfidf_afc"] = {
        "fit_seconds": time.perf_counter() - started,
        "selected_ngram_size": selection.ngram_size,
        "silhouette": selection.silhouette,
        "candidate_silhouettes": {str(key): value for key, value in selection.scores.items()},
        "clustering_adjusted_rand_index": clustering_ari,
        "vocabulary_size": len(tfidf.vectorizer_.vocabulary_),
        "initial_training_loss": float(tfidf.training_loss_[0]),
        "final_training_loss": float(tfidf.training_loss_[-1]),
        "mechanism_pass": bool(
            len(tfidf.vectorizer_.vocabulary_) > 0
            and np.isfinite(tfidf.training_loss_).all()
            and tfidf.training_loss_[-1] < tfidf.training_loss_[0]
        ),
        "kpca_fault_isolation_gate": "not_applicable_no_normal_operation_class",
    }

    print("Time-encoded histogram tests all three neural phases and learned attenuation on real event times.")
    started = time.perf_counter()
    train_timed = [timed_sequence(episode) for episode in train_episodes]
    if any(sequence is None for sequence in train_timed):
        raise RuntimeError("training contains empty activation sequences")
    histogram = OptimalTimeEncodedHistogramClassifier(
        random_state=seed,
        **parameters["time_encoded_histogram_hybrid"],
    ).fit(train_timed, y_train)
    models["time_encoded_histogram_hybrid"] = histogram
    history = histogram.training_history_
    diagnostics["time_encoded_histogram_hybrid"] = {
        "fit_seconds": time.perf_counter() - started,
        "attenuation": histogram.attenuation_,
        "vocabulary_size": len(histogram.tag_vocabulary_),
        "initial_losses": {key: float(value[0]) for key, value in history.items()},
        "final_losses": {key: float(value[-1]) for key, value in history.items()},
        "mechanism_pass": bool(
            np.isfinite(histogram.attenuation_)
            and histogram.attenuation_ > 0
            and all(np.isfinite(value).all() for value in history.values())
        ),
    }
    return models, diagnostics


def evaluate_classification(
    bundle: dict[str, Any],
    models: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    z_threshold: float,
) -> dict[str, Any]:
    truth = bundle["y"]["test"]
    episodes = bundle["episodes"]["test"]
    records: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_id in POINT_MODEL_IDS:
        started = time.perf_counter()
        predicted = predict_point_model(model_id, models[model_id], episodes, bundle)
        prediction_seconds = time.perf_counter() - started
        if predicted.shape != truth.shape or not set(predicted.tolist()) <= set(truth.tolist()):
            raise RuntimeError(f"{model_id} emitted invalid closed-set labels")
        predictions[model_id] = predicted
        metrics = multiclass_classification_metrics(truth.tolist(), predicted.tolist())
        predicted_classes = len(set(predicted.tolist()))
        chance = 1.0 / len(set(truth.tolist()))
        records[model_id] = {
            "metrics": metrics,
            "prediction_seconds": prediction_seconds,
            "predicted_classes": predicted_classes,
            "correct_by_episode": (predicted == truth).astype(int).tolist(),
            "diagnostics": diagnostics[model_id],
            "gates": {
                "valid": True,
                "activation": bool(
                    diagnostics[model_id]["mechanism_pass"] and predicted_classes >= 2
                ),
                "performance": bool(metrics["balanced_accuracy"] > chance),
                "chance_balanced_accuracy": chance,
            },
        }
    parent = (predictions["jaccard_class_core"] == truth).astype(float)
    for model_id in POINT_MODEL_IDS:
        if model_id == "jaccard_class_core":
            records[model_id]["gates"]["competitive"] = None
            records[model_id]["competitive_vs_jaccard"] = None
        else:
            gate = paired_gate(
                (predictions[model_id] == truth).astype(float), parent, z_threshold
            )
            records[model_id]["gates"]["competitive"] = gate["passed"]
            records[model_id]["competitive_vs_jaccard"] = gate
    return records


def robustness_subset(bundle: dict[str, Any]) -> tuple[AlarmEpisode, ...]:
    per_class = bundle["robustness_test_per_class"]
    selected: list[AlarmEpisode] = []
    counts: Counter[str] = Counter()
    for episode in bundle["episodes"]["test"]:
        label = str(episode.label)
        if counts[label] < per_class:
            selected.append(episode)
            counts[label] += 1
    if set(counts) != set(bundle["y"]["test"].tolist()):
        raise RuntimeError("robustness subset lost a test class")
    return tuple(selected)


def balanced_score(truth: Sequence[str], predicted: Sequence[str]) -> float:
    return float(
        multiclass_classification_metrics(list(truth), list(predicted))["balanced_accuracy"]
    )


def evaluate_robustness(
    bundle: dict[str, Any],
    models: dict[str, Any],
    configuration: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    specification = configuration["robustness"]
    scenarios = tuple(
        PerturbationScenario(kind, float(severity))
        for kind in specification["perturbations"]
        for severity in specification["normalized_severity"]
    )
    draws = int(specification["monte_carlo_draws_per_outer_seed"])
    mc_seeds = tuple(seed * 100 + index for index in range(draws))
    samples = robustness_subset(bundle)
    reports: dict[str, dict[str, Any]] = {}
    point_vectors: dict[str, np.ndarray] = {}
    for model_id in POINT_MODEL_IDS:
        print(
            f"Robustness {model_id} tests missing, spurious, timing, delay, and mixed corruptions on test-only episodes."
        )
        predictor: Callable[[Sequence[AlarmEpisode]], Sequence[str]] = (
            lambda episodes, identifier=model_id: predict_point_model(
                identifier, models[identifier], episodes, bundle
            ).tolist()
        )
        started = time.perf_counter()
        report = run_afc_robustness_benchmark(
            samples,
            predictor,
            scenarios=scenarios,
            observation_progress=tuple(specification["observation_progress"]),
            seeds=mc_seeds,
            score=balanced_score,
            spurious_tags=bundle["alarm_names"],
        )
        payload = report.as_dict()
        full_auc = {
            key: value
            for key, value in report.normalized_robustness_auc.items()
            if key.endswith("@1")
        }
        point_vectors[model_id] = np.asarray(
            [point.mean_score for point in report.points], dtype=float
        )
        chance = 1.0 / len(set(bundle["y"]["test"].tolist()))
        reports[model_id] = {
            "report": payload,
            "wall_clock_seconds": time.perf_counter() - started,
            "full_progress_mean_normalized_auc": float(np.mean(list(full_auc.values()))),
            "full_progress_auc_by_perturbation": full_auc,
            "maximum_degradation": float(max(point.degradation for point in report.points)),
            "gates": {
                "valid": bool(
                    len(report.points)
                    == len(scenarios) * len(specification["observation_progress"])
                    and np.isfinite(point_vectors[model_id]).all()
                ),
                "activation": bool(len(set(predictor(samples))) >= 2),
                "performance": bool(np.mean(list(full_auc.values())) > chance),
                "chance_balanced_accuracy": chance,
            },
        }
    parent = point_vectors["jaccard_class_core"]
    z_threshold = float(configuration["gates"]["paired_z"])
    for model_id in POINT_MODEL_IDS:
        if model_id == "jaccard_class_core":
            reports[model_id]["gates"]["competitive"] = None
            reports[model_id]["competitive_vs_jaccard"] = None
        else:
            gate = paired_gate(point_vectors[model_id], parent, z_threshold)
            reports[model_id]["gates"]["competitive"] = gate["passed"]
            reports[model_id]["competitive_vs_jaccard"] = gate
    return {
        "sample_ids": [episode.episode_id for episode in samples],
        "sample_class_counts": dict(sorted(Counter(str(row.label) for row in samples).items())),
        "scenario_count": len(scenarios),
        "monte_carlo_seeds": list(mc_seeds),
        "models": reports,
        "reporting_boundary": specification["reporting_boundary"],
    }


def prefix_lengths(bundle: dict[str, Any], fractions: Sequence[float], minimum: int = 5) -> tuple[int, ...]:
    values = {
        min(bundle["length"], max(minimum, int(round(bundle["length"] * float(value)))))
        for value in fractions
    }
    values.add(bundle["length"])
    return tuple(sorted(values))


def split_conformal_calibration(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = bundle["y"]["calibration"]
    conformal, forecast = [], []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        middle = len(indices) // 2
        if middle < 1 or len(indices) - middle < 1:
            raise RuntimeError("calibration partition cannot be split by class")
        conformal.extend(indices[:middle].tolist())
        forecast.extend(indices[middle:].tolist())
    return np.asarray(sorted(conformal)), np.asarray(sorted(forecast))


def set_metric_dict(metrics: Any) -> dict[str, float | None]:
    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in asdict(metrics).items()
    }


def stratified_folds(labels: np.ndarray, folds: int) -> np.ndarray:
    assignments = np.empty(len(labels), dtype=int)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) < folds:
            raise RuntimeError("cross-conformal class has fewer rows than folds")
        assignments[indices] = np.arange(len(indices)) % folds
    return assignments


def evaluate_conformal_and_forecasting(
    bundle: dict[str, Any], configuration: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    ctfh_parameters = configuration["classification"]["ctfh_fingerprinting"]
    conformal_parameters = configuration["conformal"]
    lengths = prefix_lengths(bundle, configuration["prefix_fractions"])
    X_train, y_train = bundle["X"]["train"], bundle["y"]["train"]
    X_cal, y_cal = bundle["X"]["calibration"], bundle["y"]["calibration"]
    X_test, y_test = bundle["X"]["test"], bundle["y"]["test"]
    conformal_indices, forecast_indices = split_conformal_calibration(bundle)

    print("ConE tests class-conditional coverage and set efficiency across eight expanding prefixes.")
    started = time.perf_counter()
    cone_models = {
        length: CTFHAlarmFloodClassifier(**ctfh_parameters).fit(
            X_train[:, :, :length], y_train
        )
        for length in lengths
    }
    cone = ConEAlarmFloodClassifier(
        cone_models,
        error_rate=float(conformal_parameters["error_rate"]),
        score_kind="probability",
    ).calibrate(X_cal[conformal_indices], y_cal[conformal_indices])
    cone_metrics = {
        str(length): set_metric_dict(metrics)
        for length, metrics in cone.evaluate_evolution(X_test, y_test).items()
    }
    forecast_evolution = cone.predict_evolution(X_cal[forecast_indices])
    test_evolution = cone.predict_evolution(X_test)
    class_count = len(set(y_train.tolist()))
    cone_activation = any(
        np.mean([len(item) for item in forecast_evolution[length]]) < class_count
        for length in lengths
    )
    cone_record = {
        "metrics_by_prefix": cone_metrics,
        "prefix_lengths": list(lengths),
        "calibration_class_counts": dict(
            sorted(Counter(y_cal[conformal_indices].tolist()).items())
        ),
        "forecast_training_class_counts": dict(
            sorted(Counter(y_cal[forecast_indices].tolist()).items())
        ),
        "wall_clock_seconds": time.perf_counter() - started,
        "gates": {
            "valid": True,
            "activation": bool(cone_activation),
            "coverage_at_full_prefix": bool(
                cone_metrics[str(lengths[-1])]["coverage"]
                >= float(configuration["gates"]["coverage_target"])
            ),
            "efficient_at_full_prefix": bool(
                cone_metrics[str(lengths[-1])]["average_set_size"] < class_count
            ),
        },
    }

    print("Cross-Conformal tests whether pooled out-of-fold calibration improves coverage without full-label sets.")
    started = time.perf_counter()
    X_pool = np.concatenate((X_train, X_cal), axis=0)
    y_pool = np.concatenate((y_train, y_cal), axis=0)
    folds = int(conformal_parameters["cross_conformal_folds"])
    fold_ids = stratified_folds(y_pool, folds)
    cross_models = {
        length: {
            fold: CTFHAlarmFloodClassifier(**ctfh_parameters).fit(
                X_pool[fold_ids != fold, :, :length], y_pool[fold_ids != fold]
            )
            for fold in range(folds)
        }
        for length in lengths
    }
    cross = CrossConformalAlarmFloodClassifier(
        cross_models,
        error_rate=float(conformal_parameters["error_rate"]),
        score_kind="probability",
        class_conditional=bool(conformal_parameters["class_conditional"]),
        empty_set_policy=str(conformal_parameters["empty_set_policy"]),
    ).calibrate(X_pool, y_pool, fold_ids)
    cross_metrics = {
        str(length): set_metric_dict(metrics)
        for length, metrics in cross.evaluate_evolution(X_test, y_test).items()
    }
    cross_record = {
        "metrics_by_prefix": cross_metrics,
        "prefix_lengths": list(lengths),
        "folds": folds,
        "fold_class_counts": {
            str(fold): dict(sorted(Counter(y_pool[fold_ids == fold].tolist()).items()))
            for fold in range(folds)
        },
        "wall_clock_seconds": time.perf_counter() - started,
        "gates": {
            "valid": True,
            "activation": any(
                row["average_set_size"] < class_count for row in cross_metrics.values()
            ),
            "coverage_at_full_prefix": bool(
                cross_metrics[str(lengths[-1])]["coverage"]
                >= float(configuration["gates"]["coverage_target"])
            ),
            "efficient_at_full_prefix": bool(
                cross_metrics[str(lengths[-1])]["average_set_size"] < class_count
            ),
        },
    }

    conformal = {
        "cone_afc_2024": cone_record,
        "cross_conformal_afc_2025": cross_record,
        "reporting_boundary": (
            "CTFH is a shared independent base model. Coverage and efficiency are real-data transfer evidence, not exact paper-table reproduction."
        ),
    }
    forecasting = evaluate_uncertainty_reduction(
        forecast_evolution,
        test_evolution,
        lengths,
        bundle,
        configuration,
        seed,
    )
    return conformal, forecasting


def trajectories_from_evolution(
    evolution: dict[int, list[frozenset[Any]]], lengths: Sequence[int]
) -> list[list[frozenset[Any]]]:
    sample_count = len(evolution[int(lengths[0])])
    return [
        [evolution[int(length)][sample] for length in lengths]
        for sample in range(sample_count)
    ]


def evaluate_uncertainty_reduction(
    training_evolution: dict[int, list[frozenset[Any]]],
    test_evolution: dict[int, list[frozenset[Any]]],
    lengths: Sequence[int],
    bundle: dict[str, Any],
    configuration: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    print("Uncertainty reduction tests jackknife+ forecasts of the next prediction-set contraction.")
    parameters = configuration["uncertainty_reduction"]
    times = np.asarray(lengths, dtype=float) * bundle["sample_seconds"] / 60.0
    try:
        training = extract_bifurcation_training_data(
            trajectories_from_evolution(training_evolution, lengths), times
        )
        testing = extract_bifurcation_training_data(
            trajectories_from_evolution(test_evolution, lengths), times
        )
    except ValueError as exc:
        return {
            "status": "mechanism_inactive",
            "reason": str(exc),
            "training_rows": 0,
            "test_rows": 0,
            "gates": {"valid": True, "activation": False, "performance": False, "competitive": False},
            "reporting_boundary": "No future set-size reduction exists, so inventing regression targets is prohibited.",
        }
    rng = np.random.default_rng(seed)
    maximum = int(parameters["maximum_training_rows"])
    if len(training.time_to_reduction) > maximum:
        selected = np.sort(rng.choice(len(training.time_to_reduction), maximum, replace=False))
        training = type(training)(
            training.features[selected],
            training.time_to_reduction[selected],
            training.episode_index[selected],
            training.time_index[selected],
        )
    if len(training.time_to_reduction) < 3:
        return {
            "status": "insufficient_training_reductions",
            "training_rows": len(training.time_to_reduction),
            "test_rows": len(testing.time_to_reduction),
            "gates": {"valid": True, "activation": False, "performance": False, "competitive": False},
        }
    started = time.perf_counter()
    forecaster = UncertaintyReductionForecaster(
        n_estimators=int(parameters["n_estimators"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        random_state=seed,
        n_jobs=1,
    ).fit(training)
    point, intervals = forecaster.model_.predict(
        testing.features,
        error_rate=float(parameters["jackknife_plus_error_rate"]),
    )
    truth = testing.time_to_reduction
    absolute_error = np.abs(point - truth)
    coverage = (intervals[:, 0] <= truth) & (truth <= intervals[:, 1])
    widths = intervals[:, 1] - intervals[:, 0]
    baseline_point = float(np.median(training.time_to_reduction))
    baseline_error = np.abs(baseline_point - truth)
    lower = float(np.quantile(training.time_to_reduction, 0.05))
    upper = float(np.quantile(training.time_to_reduction, 0.95))
    baseline_coverage = (lower <= truth) & (truth <= upper)
    paired = paired_gate(
        -absolute_error,
        -baseline_error,
        float(configuration["gates"]["paired_z"]),
    )
    timer = BifurcationDelayTimer(
        delay=int(parameters["delay"]), tolerance=float(parameters["tolerance_minutes"])
    )
    emitted = 0
    for episode in sorted(set(testing.episode_index.tolist())):
        timer.reset()
        for row in np.flatnonzero(testing.episode_index == episode):
            forecast = BifurcationForecast(
                float(point[row]),
                float(intervals[row, 0]),
                float(intervals[row, 1]),
                float(parameters["jackknife_plus_error_rate"]),
            )
            emitted += timer.update(forecast) is not None
    return {
        "status": "executed",
        "training_rows": int(len(training.time_to_reduction)),
        "test_rows": int(len(testing.time_to_reduction)),
        "wall_clock_seconds": time.perf_counter() - started,
        "metrics": {
            "mean_absolute_error_minutes": float(np.mean(absolute_error)),
            "median_absolute_error_minutes": float(np.median(absolute_error)),
            "jackknife_plus_coverage": float(np.mean(coverage)),
            "mean_interval_width_minutes": float(np.mean(widths)),
            "median_baseline_mae_minutes": float(np.mean(baseline_error)),
            "median_baseline_coverage": float(np.mean(baseline_coverage)),
            "delay_timer_emissions": int(emitted),
        },
        "competitive_vs_median_time": paired,
        "gates": {
            "valid": bool(
                np.isfinite(point).all()
                and np.isfinite(intervals).all()
                and np.all(intervals[:, 0] <= intervals[:, 1])
            ),
            "activation": True,
            "performance": bool(np.mean(coverage) >= 0.8),
            "competitive": paired["passed"],
        },
        "reporting_boundary": (
            "The paper's official capsule, features, split, and tables are unavailable; 30 trees and at most 40 jackknife rows are a bounded engineering validation."
        ),
    }


def run_dataset(name: str, specification: dict[str, Any], configuration: dict[str, Any], seed: int) -> dict[str, Any]:
    print(
        f"Dataset {name} tests five SOTA point classifiers, two conformal variants, robustness, and uncertainty reduction under one grouped split."
    )
    started = time.perf_counter()
    bundle = load_bundle(name, specification, seed)
    prior = prior_gate(bundle)
    if not prior["passed"]:
        raise RuntimeError(f"{name} failed G0 prior gate: {prior['checks']}")
    models, diagnostics = fit_point_models(bundle, configuration, seed)
    classification = evaluate_classification(
        bundle, models, diagnostics, float(configuration["gates"]["paired_z"])
    )
    robustness = evaluate_robustness(bundle, models, configuration, seed)
    conformal, forecasting = evaluate_conformal_and_forecasting(bundle, configuration, seed)
    return {
        "match": bundle["match"],
        "protocol": bundle["protocol"],
        "boundary": bundle["boundary"],
        "prior_gate": prior,
        "classification": classification,
        "robustness": robustness,
        "conformal": conformal,
        "uncertainty_reduction": forecasting,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--config", default=CONFIG_DEFAULT, type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    run_name = out_dir.name
    if not run_name.startswith("run_") or not run_name[4:].isdigit():
        raise ValueError("out_dir basename must be run_1, run_2, or run_3")
    run_index = int(run_name[4:]) - 1
    seeds = tuple(int(value) for value in configuration["seeds"])
    if not 0 <= run_index < len(seeds):
        raise ValueError(f"run index exceeds configured seeds: {seeds}")
    seed = seeds[run_index]
    started_at = datetime.now(timezone.utc)
    before = time.perf_counter()
    data_paths = [ROOT / item["path"] for item in configuration["datasets"].values()]
    evidence = provenance(config_path, data_paths)
    results = {
        name: run_dataset(name, specification, configuration, seed)
        for name, specification in configuration["datasets"].items()
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "experiment_id": configuration["id"],
        "run_name": run_name,
        "seed": seed,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "git_revision": revision,
        "execution_provenance": evidence,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - before,
        "results": results,
        "reporting_boundary": configuration["reporting_boundary"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_info.json"
    final_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "run": run_name,
        "seed": seed,
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "classification_balanced_accuracy": {
            dataset: {
                model: row["metrics"]["balanced_accuracy"]
                for model, row in result["classification"].items()
            }
            for dataset, result in results.items()
        },
        "forecast_status": {
            dataset: result["uncertainty_reduction"]["status"]
            for dataset, result in results.items()
        },
        "final_info": final_path.relative_to(ROOT).as_posix(),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
