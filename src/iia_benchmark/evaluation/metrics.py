from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from iia_benchmark.models.univariate import evaluate_alarm_design


def binary_alarm_metrics(
    abnormal: Iterable[bool], alarm: Iterable[int]
) -> dict[str, float]:
    truth = np.asarray(list(abnormal), dtype=bool)
    prediction = np.asarray(list(alarm), dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("metric inputs must be non-empty and equal length")
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far, mar, aad = evaluate_alarm_design(truth, prediction)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "missed_alarm_rate": mar,
        "average_alarm_delay": aad,
    }


def root_cause_top_k_accuracy(
    truth: Sequence[str], rankings: Sequence[Sequence[str]], *, k: int = 1
) -> float:
    if k < 1 or len(truth) != len(rankings) or not truth:
        raise ValueError("truth/rankings must be non-empty, equal length, and k >= 1")
    return sum(target in ranking[:k] for target, ranking in zip(truth, rankings)) / len(truth)


def mean_reciprocal_rank(truth: Sequence[str], rankings: Sequence[Sequence[str]]) -> float:
    if len(truth) != len(rankings) or not truth:
        raise ValueError("truth and rankings must be non-empty and equal length")
    reciprocal: list[float] = []
    for target, ranking in zip(truth, rankings):
        try:
            reciprocal.append(1.0 / (list(ranking).index(target) + 1))
        except ValueError:
            reciprocal.append(0.0)
    return sum(reciprocal) / len(reciprocal)


def sequence_accuracy(truth: Sequence[str], prediction: Sequence[str]) -> float:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and equal length")
    return sum(a == b for a, b in zip(truth, prediction)) / len(truth)


def multiclass_classification_metrics(
    truth: Sequence[str], prediction: Sequence[str]
) -> dict[str, object]:
    """Return accuracy, balanced accuracy, macro F1 and an audited confusion matrix."""

    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and equal length")
    labels = tuple(sorted(set(truth) | set(prediction)))
    confusion = {
        actual: {
            predicted: sum(
                left == actual and right == predicted
                for left, right in zip(truth, prediction)
            )
            for predicted in labels
        }
        for actual in labels
    }
    recalls, f1_scores = [], []
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in labels)
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": sequence_accuracy(truth, prediction),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def prediction_set_metrics(
    truth: Sequence[str], prediction_sets: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Coverage and efficiency metrics for conformal/open-set flood classifiers."""
    if len(truth) != len(prediction_sets) or not truth:
        raise ValueError("truth and prediction_sets must be non-empty and equal length")
    coverage = sum(label in candidates for label, candidates in zip(truth, prediction_sets))
    sizes = [len(candidates) for candidates in prediction_sets]
    return {
        "empirical_coverage": coverage / len(truth),
        "mean_prediction_set_size": float(np.mean(sizes)),
        "singleton_rate": sum(size == 1 for size in sizes) / len(sizes),
    }


def robustness_degradation(clean_score: float, perturbed_score: float) -> float:
    """Absolute performance loss; negative values mean improvement under noise."""
    if not np.isfinite((clean_score, perturbed_score)).all():
        raise ValueError("scores must be finite")
    return float(clean_score - perturbed_score)
