from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull


def _as_matrix(values: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("expected at least three observations and two features")
    if not np.isfinite(matrix).all():
        raise ValueError("input contains non-finite values")
    return matrix


@dataclass
class MahalanobisAlarm:
    """Classical multivariate statistical process-control baseline."""

    quantile: float = 0.99

    def fit(self, normal_values: Iterable[Iterable[float]]) -> "MahalanobisAlarm":
        matrix = _as_matrix(normal_values)
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1")
        self.mean_ = matrix.mean(axis=0)
        self.precision_ = np.linalg.pinv(np.cov(matrix, rowvar=False))
        scores = self.score_samples(matrix)
        self.threshold_ = float(np.quantile(scores, self.quantile))
        return self

    def score_samples(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(list(values), dtype=float)
        centered = matrix - self.mean_
        return np.einsum("ij,jk,ik->i", centered, self.precision_, centered)

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        return (self.score_samples(values) > self.threshold_).astype(np.int8)


@dataclass
class ConvexHullNOZAlarm:
    """Book-inspired convex normal-operating-zone (NOZ) alarm generator."""

    false_alarm_fraction: float = 0.01

    def fit(self, normal_values: Iterable[Iterable[float]]) -> "ConvexHullNOZAlarm":
        matrix = _as_matrix(normal_values)
        if not 0 <= self.false_alarm_fraction < 0.5:
            raise ValueError("false_alarm_fraction must be in [0, 0.5)")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0, ddof=1)
        self.scale_[self.scale_ == 0] = 1.0
        normalized = (matrix - self.mean_) / self.scale_
        center = np.median(normalized, axis=0)
        distance = np.linalg.norm(normalized - center, axis=1)
        keep = max(normalized.shape[1] + 1, int(round(len(normalized) * (1 - self.false_alarm_fraction))))
        retained = normalized[np.argsort(distance)[:keep]]
        self.hull_ = ConvexHull(retained, qhull_options="QJ")
        self.equations_ = self.hull_.equations.copy()
        return self

    def decision_function(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "equations_"):
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(list(values), dtype=float)
        normalized = (matrix - self.mean_) / self.scale_
        normals = self.equations_[:, :-1]
        offsets = self.equations_[:, -1]
        return np.max(normalized @ normals.T + offsets, axis=1)

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        return (self.decision_function(values) > 1e-10).astype(np.int8)

    def dynamic_bounds(self, value: Iterable[float]) -> np.ndarray:
        """Return low/high limits for every feature conditional on the others."""
        if not hasattr(self, "equations_"):
            raise RuntimeError("model is not fitted")
        point = (np.asarray(list(value), dtype=float) - self.mean_) / self.scale_
        if point.shape != self.mean_.shape:
            raise ValueError("value has the wrong number of features")
        bounds = np.empty((len(point), 2), dtype=float)
        normals = self.equations_[:, :-1]
        offsets = self.equations_[:, -1]
        for feature in range(len(point)):
            coefficients = normals[:, feature]
            remainder = normals @ point + offsets - coefficients * point[feature]
            candidates = -remainder[np.abs(coefficients) > 1e-12] / coefficients[np.abs(coefficients) > 1e-12]
            active_coefficients = coefficients[np.abs(coefficients) > 1e-12]
            lower_values = candidates[active_coefficients < 0]
            upper_values = candidates[active_coefficients > 0]
            low = np.max(lower_values) if len(lower_values) else -np.inf
            high = np.min(upper_values) if len(upper_values) else np.inf
            bounds[feature] = (low * self.scale_[feature] + self.mean_[feature], high * self.scale_[feature] + self.mean_[feature])
        return bounds

