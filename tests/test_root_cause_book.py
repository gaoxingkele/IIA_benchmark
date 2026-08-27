import numpy as np

from iia_benchmark.models import (
    NormalizedTransferEntropyGraph,
    PLRContributionRCA,
    RecursiveBayesianAlarmRCA,
    information_granulation_direct_transfer_entropy,
    information_granulation_transfer_entropy,
    information_granules,
    lagged_correlation_delay,
    normalized_direct_transfer_entropy,
    normalized_transfer_entropy,
    piecewise_linear_representation,
)


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
    assert 0 <= igdte <= 1
    assert 0 < igte <= 1


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
