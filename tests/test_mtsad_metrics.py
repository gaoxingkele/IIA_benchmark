from __future__ import annotations

import numpy as np
import pytest

from iia_benchmark.evaluation.mtsad_metrics import (
    adjusted_metrics,
    anomaly_ranges,
    best_f1_threshold,
    evaluate_scores,
    percentile_threshold,
    point_adjust,
    pointwise_metrics,
    range_based_metrics,
)


def test_point_adjust_extends_a_detected_segment() -> None:
    truth = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    prediction = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    adjusted = point_adjust(truth, prediction)
    assert adjusted.tolist() == [False, True, True, True, False, False, False, False, False]
    # The undetected second segment must stay undetected.
    assert adjusted_metrics(truth, prediction)["recall"] == pytest.approx(3 / 5)
    assert pointwise_metrics(truth, prediction)["recall"] == pytest.approx(1 / 5)


def test_point_adjust_is_a_no_op_without_detections() -> None:
    truth = np.array([0, 1, 1, 0], dtype=bool)
    prediction = np.zeros(4, dtype=bool)
    assert not point_adjust(truth, prediction).any()


def test_anomaly_ranges_are_inclusive() -> None:
    assert anomaly_ranges(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert anomaly_ranges(np.zeros(3, dtype=bool)) == []


def test_pointwise_metrics_are_exact() -> None:
    truth = np.array([1, 1, 0, 0], dtype=bool)
    prediction = np.array([1, 0, 1, 0], dtype=bool)
    metrics = pointwise_metrics(truth, prediction)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["f1"] == pytest.approx(0.5)


def test_range_based_metric_rewards_covering_a_range() -> None:
    truth = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    covered = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    partial = np.array([0, 1, 0, 0, 0, 0], dtype=bool)
    assert range_based_metrics(truth, covered)["f1"] == pytest.approx(1.0)
    assert range_based_metrics(truth, partial)["recall"] == pytest.approx(0.25)


def test_best_f1_threshold_matches_a_brute_force_search() -> None:
    rng = np.random.default_rng(0)
    truth = rng.random(2000) < 0.1
    scores = rng.normal(size=2000)
    scores[truth] += 2.0
    found = best_f1_threshold(truth, scores)
    brute = 0.0
    for threshold in np.unique(scores):
        brute = max(brute, pointwise_metrics(truth, scores > threshold)["f1"])
    assert found["f1"] == pytest.approx(brute)


def test_percentile_threshold_uses_both_splits() -> None:
    train = np.zeros(100)
    test = np.ones(100)
    # Half of the concatenated pool is zero and half is one, so the 50th
    # percentile sits exactly on the step.
    threshold = percentile_threshold(train, test, anomaly_ratio=50.0)
    assert threshold == pytest.approx(0.5)
    assert percentile_threshold(train, test, anomaly_ratio=1.0) == pytest.approx(1.0)


def test_evaluate_scores_reports_both_families() -> None:
    truth = np.array([0, 1, 1, 0, 0], dtype=bool)
    scores = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    result = evaluate_scores(truth=truth, scores=scores, threshold=0.5)
    assert result["pointwise_f1"] == pytest.approx(2 / 3)
    assert result["adjusted_f1"] == pytest.approx(1.0)
    assert result["reported_protocol_f1"] == result["adjusted_f1"]
    strict = evaluate_scores(
        truth=truth, scores=scores, threshold=0.5, point_adjustment=False
    )
    assert strict["reported_protocol_f1"] == strict["pointwise_f1"]


def test_evaluate_scores_ignores_unscored_points() -> None:
    truth = np.array([1, 1, 1, 0], dtype=bool)
    scores = np.array([np.nan, 2.0, 2.0, 0.0])
    result = evaluate_scores(truth=truth, scores=scores, threshold=1.0)
    assert result["pointwise_f1"] == pytest.approx(1.0)
    assert result["scored_points"] == 3
    assert result["unscored_points"] == 1
