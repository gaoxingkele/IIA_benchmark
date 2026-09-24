from __future__ import annotations

import time

import numpy as np

from iia_benchmark.evaluation.mtsad_metrics import (
    anomaly_ranges,
    point_adjust,
    range_based_metrics,
)


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
