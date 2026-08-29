from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from iia_benchmark.models import (
    LinearSegment,
    NormalizedTransferEntropyGraph,
    PLRContributionRCA,
    RecursiveBayesianAlarmRCA,
    clustered_surrogate_threshold,
    discrete_direct_transfer_entropy,
    discrete_transfer_entropy,
    information_granulation_direct_transfer_entropy,
    information_granulation_transfer_entropy,
    information_granules,
    lagged_correlation_delay,
    normalized_direct_transfer_entropy,
    normalized_transfer_entropy,
    piecewise_lagged_correlation_delay,
    piecewise_linear_representation,
)
from iia_benchmark.runner import _run_real_alarm_causal_graph


def _binary_chain(length: int = 2500, lag: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    root = rng.binomial(1, 0.08, length)
    middle = np.zeros(length, dtype=int)
    target = np.zeros(length, dtype=int)
    middle[lag:] = root[:-lag]
    target[lag:] = middle[:-lag]
    middle = np.maximum(middle, rng.binomial(1, 0.005, length))
    target = np.maximum(target, rng.binomial(1, 0.005, length))
    return root, middle, target


def test_nte_and_ndte_remove_indirect_edge() -> None:
    root, middle, target = _binary_chain()
    nte = normalized_transfer_entropy(root, target, lag=6)
    ndte = normalized_direct_transfer_entropy(
        root, target, middle, source_lag=6, intermediate_lag=3
    )
    assert nte > 0.1
    assert ndte < nte
    te = discrete_transfer_entropy(root, target, lag=6)
    dte = discrete_direct_transfer_entropy(
        root, target, middle, source_lag=6, intermediate_lag=3
    )
    assert te > 0
    assert dte < te


def test_nte_graph_finds_direct_chain_edges() -> None:
    root, middle, target = _binary_chain()
    graph = NormalizedTransferEntropyGraph(
        max_lag=6, simulations=9, significance=0.1, minimum_occurrences=30, seed=3
    )
    edges = graph.infer({"root": root, "middle": middle, "target": target})
    assert any(edge.source == "root" and edge.target == "middle" for edge in edges)
    assert any(edge.source == "middle" and edge.target == "target" for edge in edges)


def test_information_granulation_te_and_direct_te() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=1200)
    middle = np.roll(source, 10) + rng.normal(0, 0.1, len(source))
    target = np.roll(middle, 10) + rng.normal(0, 0.1, len(source))
    granules = information_granules(source, 10)
    assert granules.shape == (120, 3)
    igte = information_granulation_transfer_entropy(
        source, target, window_size=10, lag=2, order=2, min_samples=4
    )
    igdte = information_granulation_direct_transfer_entropy(
        source, target, middle, window_size=10, lag=1, order=2, min_samples=4
    )
    assert igdte >= 0
    assert igte > 0
    xlabels = rng.binomial(1, 0.2, 2000)
    ylabels = np.roll(xlabels, 2)
    threshold = clustered_surrogate_threshold(
        xlabels, ylabels, order=2, simulations=5, significance=0.2, seed=3
    )
    assert discrete_transfer_entropy(
        xlabels, ylabels, lag=2, source_horizon=2, target_horizon=2
    ) > threshold
    with pytest.raises(ValueError, match="lags be non-negative"):
        discrete_direct_transfer_entropy(
            xlabels, ylabels, xlabels, source_lag=-1
        )


def test_igdte_accepts_distinct_path_lags_and_prunes_an_indirect_chain() -> None:
    root, middle, target = _binary_chain()
    rng = np.random.default_rng(17)
    continuous = [
        np.repeat(values, 5) + rng.normal(0.0, 0.01, len(values) * 5)
        for values in (root, middle, target)
    ]
    igte = information_granulation_transfer_entropy(
        continuous[0], continuous[2], window_size=5, lag=6, order=2, min_samples=4
    )
    igdte = information_granulation_direct_transfer_entropy(
        continuous[0],
        continuous[2],
        continuous[1],
        window_size=5,
        source_lag=6,
        intermediate_lag=3,
        order=2,
        min_samples=4,
    )
    assert igte > 0.1
    assert igdte < 0.2 * igte


def test_recursive_bayesian_network_converges_and_handles_unknown() -> None:
    model = RecursiveBayesianAlarmRCA(["pump", "valve"], response_time_samples=10)
    for _ in range(80):
        model.update([1, 0], 1)
    assert model.root_cause() == ("pump",)
    assert model.update_rate == 1.0 - 0.5 ** 0.1
    unknown = RecursiveBayesianAlarmRCA(["pump", "valve"], response_time_samples=5)
    for _ in range(80):
        unknown.update([0, 0], 1)
    assert unknown.root_cause() == ("unknown",)


def test_recursive_bayesian_sequence_uses_book_zero_initialization() -> None:
    model = RecursiveBayesianAlarmRCA(["cause"], response_time_samples=5, initial_probability=0)
    assert np.all(model.cause_probabilities_ == 0)
    decisions = model.infer_sequence([[0]] * 10 + [[1]] * 20, [0] * 10 + [1] * 20)
    assert decisions[-1] == ("cause",)


def test_plr_delay_and_nonnegative_contributions_rank_driver() -> None:
    rng = np.random.default_rng(4)
    x1 = np.r_[np.zeros(60), np.linspace(0, 10, 80), np.full(60, 10)]
    x2 = rng.normal(0, 0.1, len(x1))
    y = 3.0 * np.roll(x1, 4) + 0.1 * x2
    lag, correlation, threshold = lagged_correlation_delay(x1, y, max_lag=10)
    assert lag == 4
    assert abs(correlation) > threshold
    segments = piecewise_linear_representation(y, max_segments=5, min_size=15)
    assert len(segments) >= 2
    results = PLRContributionRCA(max_segments=5, min_size=15, max_lag=10).analyze(
        np.column_stack([x1, x2]), y
    )
    changing = [result for result in results if result.target_trend != 0]
    assert changing
    assert max(result.factors[0] for result in changing) > 0.8


def test_piecewise_plr_reproduces_book_numerical_delays_and_dominant_halves() -> None:
    rng = np.random.default_rng(41)
    time = np.arange(1, 3201, dtype=float)
    sine = np.sin(np.pi * time / 100.0 - np.pi / 2.0)
    x1 = np.empty(3200)
    x2 = np.empty(3200)
    x1[:1600] = 5.0 * sine[:1600] + 5.0 + rng.normal(size=1600)
    x2[:800] = 2.5 * sine[:800] + 2.5 + 1.5 * rng.normal(size=800)
    x2[800:1600] = 1.5 * rng.normal(size=800)
    x1[1600:2400] = 2.5 * sine[1600:2400] + 2.5 + 1.5 * rng.normal(size=800)
    x1[2400:] = 1.5 * rng.normal(size=800)
    x2[1600:] = 5.0 * sine[1600:] + 5.0 + rng.normal(size=1600)
    y = np.zeros(3200)
    for index in range(3200):
        x1_delayed = x1[max(index - 10, 0)]
        x2_delayed = x2[max(index - 8, 0)]
        if index < 800:
            y[index] = 1.5 * x1_delayed + x2_delayed
        elif index < 1600:
            y[index] = 1.5 * x1_delayed
        elif index < 2400:
            y[index] = x1_delayed + 1.5 * x2_delayed
        else:
            y[index] = 1.5 * x2_delayed
    y += rng.normal(size=3200)
    boundaries = tuple(range(0, 3201, 100))
    fixed_segments = [
        LinearSegment(start, stop, 0.0, 0.0, 0.0, 0.0)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    delay1, evidence1 = piecewise_lagged_correlation_delay(
        x1, y, fixed_segments, max_lag=15
    )
    delay2, evidence2 = piecewise_lagged_correlation_delay(
        x2, y, fixed_segments, max_lag=15
    )
    assert delay1 == 10
    assert delay2 == 8
    assert sum(bool(row["significant"]) for row in evidence1) >= 20
    assert sum(bool(row["significant"]) for row in evidence2) >= 20
    results = PLRContributionRCA(max_segments=32, min_size=80, max_lag=15).analyze(
        np.column_stack([x1, x2]), y, segment_boundaries=boundaries
    )
    first_half = [result for result in results[:16] if result.target_trend != 0]
    second_half = [result for result in results[16:] if result.target_trend != 0]
    assert np.mean([result.factors[0] > result.factors[1] for result in first_half]) >= 0.75
    assert np.mean([result.factors[1] > result.factors[0] for result in second_half]) >= 0.75


def test_grouped_alarm_nte_runner_activates_and_prunes(tmp_path: Path) -> None:
    archive = tmp_path / "alarms.zip"
    with ZipFile(archive, "w") as destination:
        for label in ("A", "B"):
            for run_number in (1, 2):
                root, middle, target = _binary_chain(length=500, lag=3)
                if label == "B":
                    root, target = target.copy(), root.copy()
                noise = np.random.default_rng(run_number).binomial(1, 0.08, len(root))
                matrix = np.column_stack([root, middle, target, noise])
                rows = ["ROOT,MIDDLE,TARGET,NOISE"] + [
                    ",".join(map(str, row)) for row in matrix
                ]
                destination.writestr(
                    f"alarmseriesdata/{label}/{label}_run{run_number}_alarm.csv",
                    "\n".join(rows) + "\n",
                )
    result = _run_real_alarm_causal_graph(
        tmp_path,
        {
            "dataset": {
                "loader": "fcc_alarm_zip",
                "alarm_archive": "alarms.zip",
                "alarm_representation": "state",
                "scenarios": ["A", "B"],
            },
            "split": {"id": "synthetic_grouped", "test_run_numbers": [1, 2]},
            "model": {
                "analysis_runs_per_class": 2,
                "minimum_occurrences": 3,
                "minimum_clear_samples": 3,
                "max_features": 4,
                "max_lag": 6,
                "simulations": 5,
                "significance": 0.1,
                "seed": 17,
            },
        },
    )
    assert result["activation"]["passed"]
    assert result["activation"]["analysis_runs"] == 4
    assert result["activation"]["nte_significant_edges"] > 0
    assert result["activation"]["ndte_pruned_edges"] > 0
    assert result["reporting_status"].startswith("Book Chapter 4.1")
