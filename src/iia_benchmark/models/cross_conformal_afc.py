"""Data-efficient cross-conformal prediction for online alarm floods.

The ICPS 2025 paper confirms cross-conformal aggregation over early alarm-flood
classifiers and an empty-set postprocessing step.  The full text is access
controlled in the current environment, so this module implements the standard
pooled cross-conformal p-value construction and makes the empty-set policy
explicit.  Paper-specific scores and tables remain a separate closure gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from .cone_afc import ConformalSetMetrics, evaluate_prediction_sets


@dataclass(frozen=True)
class CrossConformalDiagnostics:
    """Frozen calibration metadata used to audit a cross-conformal run."""

    classes: tuple[Hashable, ...]
    windows: tuple[int, ...]
    folds: tuple[Hashable, ...]
    error_rate: float
    score_kind: str
    class_conditional: bool
    calibration_counts: np.ndarray


class CrossConformalAFCCalibrator:
    """Aggregate out-of-fold conformity scores into class p-values.

    ``scores_by_window[window][fold]`` contains scores produced for that fold by
    a model trained without the fold.  For probability conformity, larger test
    scores receive larger p-values; for distance nonconformity, smaller scores
    receive larger p-values.  The default is Mondrian/class-conditional
    calibration, which prevents frequent flood classes dominating rare ones.
    """

    def __init__(
        self,
        error_rate: float = 0.1,
        score_kind: str = "probability",
        class_conditional: bool = True,
        empty_set_policy: str = "top_p_value",
    ) -> None:
        if not 0 <= error_rate < 1:
            raise ValueError("error_rate must be in [0, 1)")
        if score_kind not in {"probability", "distance"}:
            raise ValueError("score_kind must be 'probability' or 'distance'")
        if empty_set_policy not in {"top_p_value", "keep_empty"}:
            raise ValueError("empty_set_policy must be 'top_p_value' or 'keep_empty'")
        self.error_rate = float(error_rate)
        self.score_kind = score_kind
        self.class_conditional = bool(class_conditional)
        self.empty_set_policy = empty_set_policy

    def fit(
        self,
        scores_by_window: Mapping[int, Mapping[Hashable, np.ndarray]],
        y_by_fold: Mapping[Hashable, np.ndarray],
        classes: tuple[Hashable, ...] | list[Hashable] | np.ndarray,
    ) -> "CrossConformalAFCCalibrator":
        class_tuple = tuple(classes)
        if not class_tuple or len(set(class_tuple)) != len(class_tuple):
            raise ValueError("classes must be nonempty and unique")
        folds = tuple(sorted(y_by_fold, key=repr))
        if len(folds) < 2:
            raise ValueError("cross-conformal calibration requires at least two folds")
        labels_by_fold = {fold: np.asarray(y_by_fold[fold]) for fold in folds}
        for fold, labels in labels_by_fold.items():
            if labels.ndim != 1 or labels.size == 0:
                raise ValueError(f"fold {fold!r} labels must be a nonempty vector")
            if not set(labels.tolist()).issubset(set(class_tuple)):
                raise ValueError(f"fold {fold!r} contains a label outside classes")

        windows = tuple(sorted(int(window) for window in scores_by_window))
        if not windows:
            raise ValueError("at least one prefix window is required")
        calibration: dict[int, dict[Hashable, tuple[np.ndarray, ...]]] = {}
        counts = np.zeros((len(windows), len(folds), len(class_tuple)), dtype=int)
        for window_index, window in enumerate(windows):
            fold_scores = scores_by_window[window]
            if set(fold_scores) != set(folds):
                raise ValueError(f"window {window} must contain exactly the calibrated folds")
            calibration[window] = {}
            for fold_index, fold in enumerate(folds):
                values = np.asarray(fold_scores[fold], dtype=float)
                labels = labels_by_fold[fold]
                if values.shape != (labels.size, len(class_tuple)):
                    raise ValueError(
                        f"window {window}, fold {fold!r} scores must have shape "
                        f"{(labels.size, len(class_tuple))}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError("calibration scores must be finite")
                true_indices = np.asarray([class_tuple.index(label) for label in labels])
                true_scores = values[np.arange(labels.size), true_indices]
                per_class: list[np.ndarray] = []
                for class_index, label in enumerate(class_tuple):
                    selected = true_scores[labels == label] if self.class_conditional else true_scores
                    per_class.append(selected.copy())
                    counts[window_index, fold_index, class_index] = selected.size
                calibration[window][fold] = tuple(per_class)

        if self.class_conditional and np.any(np.sum(counts, axis=1) == 0):
            raise ValueError("every class needs at least one out-of-fold calibration example")
        self._calibration = calibration
        self.diagnostics_ = CrossConformalDiagnostics(
            class_tuple,
            windows,
            folds,
            self.error_rate,
            self.score_kind,
            self.class_conditional,
            counts,
        )
        return self

    def predict_p_values(
        self, scores_by_fold: Mapping[Hashable, np.ndarray], window: int
    ) -> np.ndarray:
        if not hasattr(self, "diagnostics_"):
            raise RuntimeError("fit must be called before predict_p_values")
        if int(window) not in self._calibration:
            raise KeyError(f"unknown calibrated window: {window}")
        if set(scores_by_fold) != set(self.diagnostics_.folds):
            raise ValueError("test scores must contain exactly the calibrated folds")

        values_by_fold: dict[Hashable, np.ndarray] = {}
        row_count: int | None = None
        for fold in self.diagnostics_.folds:
            values = np.asarray(scores_by_fold[fold], dtype=float)
            if values.ndim != 2 or values.shape[1] != len(self.diagnostics_.classes):
                raise ValueError(f"fold {fold!r} test scores have an incompatible shape")
            if row_count is None:
                row_count = values.shape[0]
            if values.shape[0] != row_count or not np.all(np.isfinite(values)):
                raise ValueError("all fold test scores must be finite and have equal rows")
            values_by_fold[fold] = values

        p_values = np.ones((row_count or 0, len(self.diagnostics_.classes)), dtype=float)
        for row in range(row_count or 0):
            for class_index in range(len(self.diagnostics_.classes)):
                support = 1
                denominator = 1
                for fold in self.diagnostics_.folds:
                    calibration = self._calibration[int(window)][fold][class_index]
                    test_score = values_by_fold[fold][row, class_index]
                    if self.score_kind == "probability":
                        support += int(np.count_nonzero(calibration <= test_score))
                    else:
                        support += int(np.count_nonzero(calibration >= test_score))
                    denominator += calibration.size
                p_values[row, class_index] = support / denominator
        return p_values

    def prediction_sets_from_p_values(
        self, p_values: np.ndarray
    ) -> list[frozenset[Hashable]]:
        if not hasattr(self, "diagnostics_"):
            raise RuntimeError("fit must be called before prediction")
        values = np.asarray(p_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.diagnostics_.classes):
            raise ValueError("p_values have an incompatible class dimension")
        if not np.all((0 <= values) & (values <= 1)):
            raise ValueError("p_values must lie in [0, 1]")
        result: list[frozenset[Hashable]] = []
        for row in values:
            included = [
                label
                for label, p_value in zip(self.diagnostics_.classes, row)
                if p_value > self.error_rate
            ]
            if not included and self.empty_set_policy == "top_p_value":
                included = [self.diagnostics_.classes[int(np.argmax(row))]]
            result.append(frozenset(included))
        return result

    def predict_sets(
        self, scores_by_fold: Mapping[Hashable, np.ndarray], window: int
    ) -> list[frozenset[Hashable]]:
        return self.prediction_sets_from_p_values(
            self.predict_p_values(scores_by_fold, window)
        )

    def evaluate(
        self,
        scores_by_fold: Mapping[Hashable, np.ndarray],
        y_true: np.ndarray,
        window: int,
    ) -> ConformalSetMetrics:
        return evaluate_prediction_sets(self.predict_sets(scores_by_fold, window), y_true)


class CrossConformalAlarmFloodClassifier:
    """Apply fold-specific, pre-trained classifiers to expanding prefixes.

    Models are indexed as ``models[window][fold]``.  Each fold model must have
    been trained without its corresponding fold and expose ``classes_`` plus
    ``predict_proba`` (or the configured distance method).
    """

    def __init__(
        self,
        models: Mapping[int, Mapping[Hashable, object]],
        error_rate: float = 0.1,
        score_kind: str = "probability",
        distance_method: str = "predict_distance",
        class_conditional: bool = True,
        empty_set_policy: str = "top_p_value",
    ) -> None:
        if not models or any(not fold_models for fold_models in models.values()):
            raise ValueError("models must contain window and fold models")
        self.models = {
            int(window): dict(fold_models) for window, fold_models in models.items()
        }
        self.error_rate = float(error_rate)
        self.score_kind = score_kind
        self.distance_method = distance_method
        self.class_conditional = bool(class_conditional)
        self.empty_set_policy = empty_set_policy

    def _scores(self, model: object, X: np.ndarray) -> np.ndarray:
        method_name = "predict_proba" if self.score_kind == "probability" else self.distance_method
        method = getattr(model, method_name, None)
        if method is None:
            raise TypeError(f"wrapped model does not expose {method_name}()")
        return np.asarray(method(X), dtype=float)

    def calibrate(
        self, X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray
    ) -> "CrossConformalAlarmFloodClassifier":
        episodes = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        assignments = np.asarray(fold_ids)
        if episodes.ndim != 3:
            raise ValueError("X must have shape (episodes, alarm_variables, time)")
        if labels.shape != (episodes.shape[0],) or assignments.shape != labels.shape:
            raise ValueError("y and fold_ids must contain one value per episode")
        windows = tuple(sorted(self.models))
        if windows[-1] > episodes.shape[2]:
            raise ValueError("a model window exceeds the calibration episode length")
        folds = tuple(sorted(set(assignments.tolist()), key=repr))
        if set(self.models[windows[0]]) != set(folds):
            raise ValueError("model folds and fold_ids do not match")
        first_model = self.models[windows[0]][folds[0]]
        classes = tuple(getattr(first_model, "classes_", ()))
        if not classes:
            raise TypeError("wrapped models must expose nonempty classes_")

        scores_by_window: dict[int, dict[Hashable, np.ndarray]] = {}
        for window in windows:
            if set(self.models[window]) != set(folds):
                raise ValueError("all windows must contain the same folds")
            scores_by_window[window] = {}
            for fold in folds:
                model = self.models[window][fold]
                if tuple(getattr(model, "classes_", ())) != classes:
                    raise ValueError("all fold models must use the same class order")
                mask = assignments == fold
                scores_by_window[window][fold] = self._scores(
                    model, episodes[mask, :, :window]
                )
        y_by_fold = {fold: labels[assignments == fold] for fold in folds}
        self.classes_ = np.asarray(classes)
        self.calibrator_ = CrossConformalAFCCalibrator(
            self.error_rate,
            self.score_kind,
            self.class_conditional,
            self.empty_set_policy,
        ).fit(scores_by_window, y_by_fold, classes)
        return self

    def _test_scores(self, X: np.ndarray, window: int) -> dict[Hashable, np.ndarray]:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("calibrate must be called before prediction")
        if window not in self.models:
            raise KeyError(f"no models are registered for window {window}")
        episodes = np.asarray(X, dtype=float)
        if episodes.ndim != 3 or window > episodes.shape[2]:
            raise ValueError("X has an incompatible shape or prefix length")
        return {
            fold: self._scores(model, episodes[:, :, :window])
            for fold, model in self.models[window].items()
        }

    def predict_p_values(self, X: np.ndarray, window: int) -> np.ndarray:
        return self.calibrator_.predict_p_values(self._test_scores(X, window), window)

    def predict_sets(self, X: np.ndarray, window: int) -> list[frozenset[Hashable]]:
        return self.calibrator_.predict_sets(self._test_scores(X, window), window)

    def predict_evolution(self, X: np.ndarray) -> dict[int, list[frozenset[Hashable]]]:
        return {window: self.predict_sets(X, window) for window in sorted(self.models)}

    def evaluate_evolution(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> dict[int, ConformalSetMetrics]:
        return {
            window: evaluate_prediction_sets(prediction_sets, y_true)
            for window, prediction_sets in self.predict_evolution(X).items()
        }
