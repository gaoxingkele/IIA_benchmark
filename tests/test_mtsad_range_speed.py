from __future__ import annotations

import time

import numpy as np

from iia_benchmark.evaluation.mtsad_metrics import (
    anomaly_ranges,
    point_adjust,
    range_based_metrics,
)


def _naive_range_metrics(truth, prediction, *, alpha=0.0, bias="flat"):
    """Straightforward O(ranges^2) reference used to validate the sweep."""

    from iia_benchmark.evaluation.mtsad_metrics import _bias

    real = anomaly_ranges(truth)
    predicted = anomaly_ranges(prediction)
    if not real and not predicted:
        return {"precision": 1.0, "recall": 1.0}
    if not real or not predicted:
        return {"precision": 0.0, "recall": 0.0}
    precisions = []
    for pred_start, pred_end in predicted:
        best = 0.0
        for real_start, real_end in real:
            start = max(pred_start, real_start)
            end = min(pred_end, real_end)
            if end >= start:
                best = max(best, (end - start + 1) / (pred_end - pred_start + 1))
        precisions.append(best)
    recalls = []
    for real_start, real_end in real:
        real_len = real_end - real_start + 1
        existence = 0.0
        best = 0.0
        for pred_start, pred_end in predicted:
            start = max(real_start, pred_start)
            end = min(real_end, pred_end)
            if end >= start:
                existence = 1.0
                position = (start - real_start) / real_len
                overlap = (end - start + 1) / real_len
                best = max(best, _bias(overlap, real_len, position, bias))
        recalls.append(alpha * existence + (1.0 - alpha) * best)
    return {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls))}


def test_sweep_matches_the_naive_interval_loop() -> None:
    rng = np.random.default_rng(23)
    for _ in range(60):
        length = int(rng.integers(50, 4000))
        truth = rng.random(length) < rng.uniform(0.01, 0.3)
        prediction = rng.random(length) < rng.uniform(0.01, 0.3)
        expected = _naive_range_metrics(truth, prediction)
        actual = range_based_metrics(truth, prediction)
        assert actual["precision"] == expected["precision"]
        assert actual["recall"] == expected["recall"]


def test_anomaly_ranges_matches_a_reference_loop() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        flags = rng.random(500) < 0.2
        expected: list[tuple[int, int]] = []
        start = None
        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                expected.append((start, index - 1))
                start = None
        if start is not None:
            expected.append((start, len(flags) - 1))
        assert anomaly_ranges(flags) == expected


def test_anomaly_ranges_handles_empty_and_single_point() -> None:
    assert anomaly_ranges(np.zeros(10, dtype=bool)) == []
    assert anomaly_ranges(np.array([False, True, False])) == [(1, 1)]
    assert anomaly_ranges(np.ones(3, dtype=bool)) == [(0, 2)]


def test_metrics_stay_fast_on_a_multi_million_point_layout() -> None:
    """The concatenated window layout reaches millions of timestamps."""

    rng = np.random.default_rng(11)
    size = 2_000_000
    truth = np.zeros(size, dtype=bool)
    for start in range(1000, size, 20000):
        truth[start:start + 400] = True
    scores = rng.random(size)
    scores[truth] += 0.4
    prediction = scores > 0.6
    started = time.perf_counter()
    ranges = anomaly_ranges(prediction)
    adjusted = point_adjust(truth, prediction)
    metrics = range_based_metrics(truth, adjusted)
    elapsed = time.perf_counter() - started
    assert ranges
    assert 0.0 <= metrics["f1"] <= 1.0
    # A per-element Python loop would take tens of seconds here.
    assert elapsed < 20.0, f"metrics took {elapsed:.1f}s on {size} timestamps"


def test_range_metrics_stay_fast_with_many_scattered_predicted_ranges() -> None:
    """A quadratic interval comparison previously stalled on this shape."""

    rng = np.random.default_rng(13)
    size = 500_000
    truth = np.zeros(size, dtype=bool)
    for start in range(500, size, 5000):
        truth[start:start + 200] = True
    prediction = rng.random(size) < 0.05
    started = time.perf_counter()
    metrics = range_based_metrics(truth, prediction)
    elapsed = time.perf_counter() - started
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert elapsed < 10.0, f"range metrics took {elapsed:.1f}s"
