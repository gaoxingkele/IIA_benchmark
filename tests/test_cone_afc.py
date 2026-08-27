import numpy as np

from iia_benchmark.models import (
    ConEAFCCalibrator,
    ConEAlarmFloodClassifier,
    evaluate_prediction_sets,
    expected_cone_coverage,
)


def test_probability_thresholds_follow_equation_2() -> None:
    y = np.array(["A"] * 5 + ["B"] * 5)
    scores = np.array(
        [[p, 1 - p] for p in [0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.3, 0.4, 0.1, 0.05]]
    )
    model = ConEAFCCalibrator(error_rate=0.2).fit({10: scores}, y, ("A", "B"))
    # ceil(0.8 * 5) = fourth item in each descending true-class score list.
    np.testing.assert_allclose(model.thresholds_.for_window(10), [0.6, 0.7])


def test_distance_thresholds_follow_equation_3() -> None:
    y = np.array([0] * 5 + [1] * 5)
    distances = np.array(
        [[d, 1.5 - d] for d in [0.1, 0.2, 0.3, 0.4, 0.5, 1.4, 1.3, 1.2, 1.1, 1.0]]
    )
    model = ConEAFCCalibrator(error_rate=0.2, score_kind="distance").fit(
        {10: distances}, y, (0, 1)
    )
    np.testing.assert_allclose(model.thresholds_.for_window(10), [0.4, 0.4])
    assert model.predict_sets(np.array([[0.2, 0.7], [0.6, 0.2]]), 10) == [
        frozenset({0}),
        frozenset({1}),
    ]


def test_stepwise_thresholds_and_set_metrics() -> None:
    y = np.array(["A", "A", "B", "B"])
    early = np.array([[0.55, 0.45], [0.6, 0.4], [0.48, 0.52], [0.4, 0.6]])
    late = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    calibrator = ConEAFCCalibrator(error_rate=0.0).fit(
        {10: early, 20: late}, y, ("A", "B")
    )
    assert not np.allclose(
        calibrator.thresholds_.for_window(10), calibrator.thresholds_.for_window(20)
    )
    sets = calibrator.predict_sets(late, 20)
    metrics = evaluate_prediction_sets(sets, y)
    assert metrics.coverage == 1.0
    assert metrics.average_set_size == 1.0
    assert metrics.singleton_accuracy == 1.0


class _PrefixProbabilityModel:
    def __init__(self, window: int) -> None:
        self.window = window
        self.classes_ = np.array(["left", "right"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        left = np.clip(np.mean(X[:, 0], axis=1), 0.05, 0.95)
        return np.column_stack((left, 1.0 - left))


def test_method_agnostic_expanding_window_wrapper() -> None:
    X = np.zeros((8, 2, 20))
    X[:4, 0, :] = 0.9
    X[4:, 0, :] = 0.1
    y = np.array(["left"] * 4 + ["right"] * 4)
    wrapper = ConEAlarmFloodClassifier(
        {10: _PrefixProbabilityModel(10), 20: _PrefixProbabilityModel(20)}, error_rate=0.25
    ).calibrate(X, y)
    evolution = wrapper.predict_evolution(X)
    assert tuple(evolution) == (10, 20)
    assert all(label in group for label, group in zip(y, evolution[20]))


def test_expected_coverage_uses_paper_equation_1() -> None:
    assert expected_cone_coverage(0.1, 0.1, 0.1) == 0.76
