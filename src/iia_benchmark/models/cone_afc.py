"""ConE-AFC: class- and stepwise conformal alarm-flood prediction sets.

This independent implementation follows Manca, Kunze, and Fay (2024),
Algorithms 1--2 and Eqs. (1)--(5).  It accepts either posterior probabilities
(larger means more conforming) or distances (smaller means more conforming).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Hashable, Mapping

import numpy as np


@dataclass(frozen=True)
class ConformalThresholds:
    """Class-wise conformal thresholds for one or more expanding windows."""

    classes: tuple[Hashable, ...]
    windows: tuple[int, ...]
    values: np.ndarray
    error_rate: float
    score_kind: str
    calibration_counts: np.ndarray

    def for_window(self, window: int) -> np.ndarray:
        try:
            position = self.windows.index(int(window))
        except ValueError as exc:
            raise KeyError(f"unknown calibrated window: {window}") from exc
        return self.values[position].copy()


@dataclass(frozen=True)
class ConformalSetMetrics:
    coverage: float
    average_set_size: float
    empty_rate: float
    singleton_rate: float
    singleton_accuracy: float


def expected_cone_coverage(error_rate: float, delta: float, epsilon: float) -> float:
    """Return the paper's finite-calibration expected coverage, Eq. (1)."""

    if any(not 0 <= value <= 1 for value in (error_rate, delta, epsilon)):
        raise ValueError("error_rate, delta, and epsilon must be in [0, 1]")
    return float(np.clip((1.0 - 0.5 * delta) * (1.0 - error_rate - epsilon), 0.0, 1.0))


def evaluate_prediction_sets(
    prediction_sets: list[frozenset[Hashable]], y_true: np.ndarray
) -> ConformalSetMetrics:
    labels = np.asarray(y_true)
    if labels.ndim != 1 or labels.size != len(prediction_sets):
        raise ValueError("y_true must contain one label per prediction set")
    sizes = np.asarray([len(item) for item in prediction_sets], dtype=float)
    covered = np.asarray([label in item for label, item in zip(labels, prediction_sets)])
    singleton = sizes == 1
    singleton_accuracy = float(np.mean(covered[singleton])) if np.any(singleton) else float("nan")
    return ConformalSetMetrics(
        coverage=float(np.mean(covered)),
        average_set_size=float(np.mean(sizes)),
        empty_rate=float(np.mean(sizes == 0)),
        singleton_rate=float(np.mean(singleton)),
        singleton_accuracy=singleton_accuracy,
    )


class ConEAFCCalibrator:
    """Calibrate and apply the paper's class-/step-conditional thresholds."""

    def __init__(self, error_rate: float = 0.1, score_kind: str = "probability") -> None:
        if not 0 <= error_rate < 1:
            raise ValueError("error_rate must be in [0, 1)")
        if score_kind not in {"probability", "distance"}:
            raise ValueError("score_kind must be 'probability' or 'distance'")
        self.error_rate = float(error_rate)
        self.score_kind = score_kind

    def fit(
        self,
        scores_by_window: Mapping[int, np.ndarray],
        y: np.ndarray,
        classes: tuple[Hashable, ...] | list[Hashable] | np.ndarray,
    ) -> "ConEAFCCalibrator":
        labels = np.asarray(y)
        class_tuple = tuple(classes)
        if labels.ndim != 1 or not class_tuple:
            raise ValueError("y must be one-dimensional and classes must be nonempty")
        if len(set(class_tuple)) != len(class_tuple):
            raise ValueError("classes must be unique")
        windows = tuple(sorted(int(window) for window in scores_by_window))
        if not windows:
            raise ValueError("at least one score window is required")

        thresholds = np.empty((len(windows), len(class_tuple)), dtype=float)
        counts = np.empty_like(thresholds, dtype=int)
        for window_index, window in enumerate(windows):
            scores = np.asarray(scores_by_window[window], dtype=float)
            if scores.shape != (labels.size, len(class_tuple)):
                raise ValueError(
                    f"window {window} scores must have shape {(labels.size, len(class_tuple))}"
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("calibration scores must be finite")
            for class_index, label in enumerate(class_tuple):
                true_class_scores = scores[labels == label, class_index]
                n_class = true_class_scores.size
                if n_class == 0:
                    raise ValueError(f"calibration data has no samples for class {label!r}")
                counts[window_index, class_index] = n_class
                if self.score_kind == "probability":
                    ordered = np.sort(true_class_scores)[::-1]
                    paper_index = ceil((1.0 - self.error_rate) * n_class)
                else:
                    ordered = np.sort(true_class_scores)
                    paper_index = floor((1.0 - self.error_rate) * n_class)
                # The paper uses one-based indexing.  The clamp only matters
                # for tiny calibration sets and extreme alpha values.
                zero_based = min(n_class - 1, max(0, paper_index - 1))
                thresholds[window_index, class_index] = ordered[zero_based]

        self.thresholds_ = ConformalThresholds(
            class_tuple,
            windows,
            thresholds,
            self.error_rate,
            self.score_kind,
            counts,
        )
        return self

    def predict_sets(self, scores: np.ndarray, window: int) -> list[frozenset[Hashable]]:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("fit must be called before predict_sets")
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.thresholds_.classes):
            raise ValueError("scores have an incompatible class dimension")
        thresholds = self.thresholds_.for_window(window)
        included = values >= thresholds if self.score_kind == "probability" else values <= thresholds
        return [
            frozenset(label for label, keep in zip(self.thresholds_.classes, row) if keep)
            for row in included
        ]

    def evaluate(
        self, scores: np.ndarray, y_true: np.ndarray, window: int
    ) -> ConformalSetMetrics:
        return evaluate_prediction_sets(self.predict_sets(scores, window), y_true)


class ConEAlarmFloodClassifier:
    """Method-agnostic ConE-AFC wrapper around pre-trained window models.

    Each model must expose ``classes_`` and either ``predict_proba`` or the
    configured distance method.  Input episodes have shape
    ``(episodes, alarm_variables, time)`` and are sliced at each window.
    """

    def __init__(
        self,
        models: Mapping[int, object],
        error_rate: float = 0.1,
        score_kind: str = "probability",
        distance_method: str = "predict_distance",
    ) -> None:
        if not models:
            raise ValueError("models must contain at least one expanding-window model")
        self.models = {int(window): model for window, model in models.items()}
        self.error_rate = float(error_rate)
        self.score_kind = score_kind
        self.distance_method = distance_method

    def _scores(self, model: object, X: np.ndarray) -> np.ndarray:
        method_name = "predict_proba" if self.score_kind == "probability" else self.distance_method
        method = getattr(model, method_name, None)
        if method is None:
            raise TypeError(f"wrapped model does not expose {method_name}()")
        return np.asarray(method(X), dtype=float)

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> "ConEAlarmFloodClassifier":
        episodes = np.asarray(X, dtype=float)
        if episodes.ndim != 3:
            raise ValueError("X must have shape (episodes, alarm_variables, time)")
        windows = tuple(sorted(self.models))
        if windows[-1] > episodes.shape[2]:
            raise ValueError("a model window exceeds the calibration episode length")
        first_classes = tuple(getattr(self.models[windows[0]], "classes_", ()))
        if not first_classes:
            raise TypeError("wrapped models must expose nonempty classes_")
        scores_by_window: dict[int, np.ndarray] = {}
        for window in windows:
            model = self.models[window]
            if tuple(getattr(model, "classes_", ())) != first_classes:
                raise ValueError("all window models must use the same class order")
            scores_by_window[window] = self._scores(model, episodes[:, :, :window])
        self.classes_ = np.asarray(first_classes)
        self.calibrator_ = ConEAFCCalibrator(self.error_rate, self.score_kind).fit(
            scores_by_window, y, first_classes
        )
        return self

    @property
    def thresholds_(self) -> ConformalThresholds:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("calibrate must be called before accessing thresholds")
        return self.calibrator_.thresholds_

    def predict_sets(self, X: np.ndarray, window: int) -> list[frozenset[Hashable]]:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("calibrate must be called before predict_sets")
        episodes = np.asarray(X, dtype=float)
        if window not in self.models:
            raise KeyError(f"no model is registered for window {window}")
        scores = self._scores(self.models[window], episodes[:, :, :window])
        return self.calibrator_.predict_sets(scores, window)

    def predict_evolution(self, X: np.ndarray) -> dict[int, list[frozenset[Hashable]]]:
        return {window: self.predict_sets(X, window) for window in sorted(self.models)}

    def evaluate_evolution(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> dict[int, ConformalSetMetrics]:
        return {
            window: evaluate_prediction_sets(sets, y_true)
            for window, sets in self.predict_evolution(X).items()
        }
