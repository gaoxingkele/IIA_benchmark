import numpy as np

from iia_benchmark.models import (
    CrossConformalAFCCalibrator,
    CrossConformalAlarmFloodClassifier,
)


def _probability_calibration():
    scores = {
        10: {
            0: np.array([[0.8, 0.2], [0.3, 0.7]]),
            1: np.array([[0.6, 0.4], [0.1, 0.9]]),
        }
    }
    labels = {0: np.array(["A", "B"]), 1: np.array(["A", "B"])}
    return scores, labels


def test_pooled_cross_conformal_probability_p_values() -> None:
    scores, labels = _probability_calibration()
    calibrator = CrossConformalAFCCalibrator(error_rate=0.5).fit(
        scores, labels, ("A", "B")
    )
    test_scores = {0: np.array([[0.7, 0.3]]), 1: np.array([[0.75, 0.25]])}
    np.testing.assert_allclose(
        calibrator.predict_p_values(test_scores, 10), [[2 / 3, 1 / 3]]
    )
    assert calibrator.predict_sets(test_scores, 10) == [frozenset({"A"})]


def test_distance_scores_use_reverse_comparison() -> None:
    scores = {
        10: {
            0: np.array([[0.2, 0.8], [0.7, 0.3]]),
            1: np.array([[0.4, 0.6], [0.9, 0.1]]),
        }
    }
    labels = {0: np.array(["A", "B"]), 1: np.array(["A", "B"])}
    calibrator = CrossConformalAFCCalibrator(
        error_rate=0.5, score_kind="distance"
    ).fit(scores, labels, ("A", "B"))
    test_scores = {0: np.array([[0.3, 0.6]]), 1: np.array([[0.2, 0.8]])}
    np.testing.assert_allclose(
        calibrator.predict_p_values(test_scores, 10), [[2 / 3, 1 / 3]]
    )


def test_empty_set_postprocessing_is_explicit() -> None:
    scores, labels = _probability_calibration()
    p_values = np.array([[0.4, 0.3]])
    repaired = CrossConformalAFCCalibrator(
        error_rate=0.8, empty_set_policy="top_p_value"
    ).fit(scores, labels, ("A", "B"))
    retained = CrossConformalAFCCalibrator(
        error_rate=0.8, empty_set_policy="keep_empty"
    ).fit(scores, labels, ("A", "B"))
    assert repaired.prediction_sets_from_p_values(p_values) == [frozenset({"A"})]
    assert retained.prediction_sets_from_p_values(p_values) == [frozenset()]


class _FoldPrefixModel:
    classes_ = np.array(["left", "right"])

    def __init__(self, adjustment: float) -> None:
        self.adjustment = adjustment

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        left = np.clip(np.mean(X[:, 0], axis=1) + self.adjustment, 0.01, 0.99)
        return np.column_stack((left, 1.0 - left))


def test_expanding_window_cross_conformal_wrapper() -> None:
    X = np.zeros((8, 2, 20))
    X[:4, 0, :] = 0.9
    X[4:, 0, :] = 0.1
    y = np.array(["left"] * 4 + ["right"] * 4)
    folds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    models = {
        window: {0: _FoldPrefixModel(0.0), 1: _FoldPrefixModel(0.01)}
        for window in (10, 20)
    }
    classifier = CrossConformalAlarmFloodClassifier(
        models, error_rate=0.4
    ).calibrate(X, y, folds)
    evolution = classifier.predict_evolution(X)
    assert tuple(evolution) == (10, 20)
    assert classifier.predict_p_values(X, 20).shape == (8, 2)
    assert classifier.evaluate_evolution(X, y)[20].coverage == 1.0


def test_every_class_requires_calibration_support() -> None:
    scores = {10: {0: np.array([[0.8, 0.2]]), 1: np.array([[0.7, 0.3]])}}
    labels = {0: np.array(["A"]), 1: np.array(["A"])}
    try:
        CrossConformalAFCCalibrator().fit(scores, labels, ("A", "B"))
    except ValueError as error:
        assert "every class" in str(error)
    else:
        raise AssertionError("class-conditional calibration must reject unsupported classes")
