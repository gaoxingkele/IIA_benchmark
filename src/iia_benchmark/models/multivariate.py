from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
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


@dataclass(frozen=True)
class ConvexHullFitness:
    """Book equation (3.8) grid-occupancy fitness for a convex NOZ."""

    inside_points: int
    counting_points: int
    fitness: float


def convex_hull_fitness_index(
    normalized_values: Iterable[Iterable[float]],
    intervals: Iterable[float],
    *,
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
    maximum_grid_points: int = 1_000_000,
) -> ConvexHullFitness:
    """Evaluate the grid fitness in Book equation (3.8).

    Values are expected to be normalized already, matching Algorithm 1 in
    Section 3.1.2. Explicit bounds make rounded textbook examples
    reproducible; otherwise each feature's sample extrema are used.
    """

    matrix = _as_matrix(normalized_values)
    delta = np.asarray(list(intervals), dtype=float)
    if delta.shape != (matrix.shape[1],) or np.any(delta <= 0):
        raise ValueError("intervals must contain one positive value per feature")
    low = np.min(matrix, axis=0) if lower is None else np.asarray(list(lower), dtype=float)
    high = np.max(matrix, axis=0) if upper is None else np.asarray(list(upper), dtype=float)
    if low.shape != delta.shape or high.shape != delta.shape or np.any(high <= low):
        raise ValueError("lower and upper bounds are invalid")
    counts = np.floor((high - low) / delta + 1e-12).astype(int)
    if np.any(counts < 1) or int(np.prod(counts, dtype=np.int64)) > maximum_grid_points:
        raise ValueError("grid is empty or exceeds maximum_grid_points")
    centers = np.asarray(
        [
            [low[j] + (index[j] + 0.5) * delta[j] for j in range(len(delta))]
            for index in product(*(range(int(count)) for count in counts))
        ],
        dtype=float,
    )
    hull = ConvexHull(matrix, qhull_options="QJ")
    inside = np.all(
        centers @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-9,
        axis=1,
    )
    occupied = np.any(
        np.all(
            np.abs(centers[:, None, :] - matrix[None, :, :])
            <= delta[None, None, :] / 2 + 1e-12,
            axis=2,
        ),
        axis=1,
    )
    inside_points = int(np.sum(inside))
    counting_points = int(np.sum(inside & occupied))
    return ConvexHullFitness(
        inside_points,
        counting_points,
        float(counting_points / inside_points) if inside_points else 0.0,
    )


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

    def closest_normal_point(self, value: Iterable[float]) -> np.ndarray:
        """Project an outside point onto the convex NOZ (Book equation 3.15)."""

        if not hasattr(self, "equations_"):
            raise RuntimeError("model is not fitted")
        raw = np.asarray(list(value), dtype=float)
        if raw.shape != self.mean_.shape or not np.isfinite(raw).all():
            raise ValueError("value has the wrong number of finite features")
        point = (raw - self.mean_) / self.scale_
        normals = self.equations_[:, :-1]
        offsets = self.equations_[:, -1]
        if np.max(normals @ point + offsets) <= 1e-10:
            return raw.copy()
        result = minimize(
            lambda candidate: 0.5 * float(np.sum((candidate - point) ** 2)),
            point,
            jac=lambda candidate: candidate - point,
            constraints={
                "type": "ineq",
                "fun": lambda candidate: -(normals @ candidate + offsets),
                "jac": lambda candidate: -normals,
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not result.success or np.max(normals @ result.x + offsets) > 1e-7:
            raise RuntimeError(f"convex NOZ projection failed: {result.message}")
        return result.x * self.scale_ + self.mean_

    def dynamic_bounds(self, value: Iterable[float]) -> np.ndarray:
        """Return low/high limits for every feature conditional on the others."""
        if not hasattr(self, "equations_"):
            raise RuntimeError("model is not fitted")
        reference = self.closest_normal_point(value)
        point = (reference - self.mean_) / self.scale_
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
