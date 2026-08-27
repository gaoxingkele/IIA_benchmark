"""Forecast ConE-AFC uncertainty-reduction (bifurcation) points.

Independent reproduction of the architecture reported by Manca, Kunze, and
Fay (2025): random-forest regression, leave-one-out jackknife+ prediction
intervals, and a delay timer that suppresses unstable online forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class BifurcationForecast:
    point: float
    lower: float
    upper: float
    error_rate: float


@dataclass(frozen=True)
class BifurcationTrainingData:
    features: np.ndarray
    time_to_reduction: np.ndarray
    episode_index: np.ndarray
    time_index: np.ndarray


def extract_bifurcation_training_data(
    trajectories: Sequence[Sequence[Sequence[Hashable]]],
    times: Sequence[float],
) -> BifurcationTrainingData:
    """Turn prediction-set trajectories into next-reduction regression rows.

    Features describe the observable history only: normalized time, current set
    size, elapsed time since the previous reduction, and reductions seen so far.
    Rows after the final reduction are right-censored and intentionally excluded.
    """

    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1 or time_values.size < 2 or np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be a strictly increasing one-dimensional sequence")
    rows: list[list[float]] = []
    targets: list[float] = []
    episode_ids: list[int] = []
    time_ids: list[int] = []
    span = time_values[-1] - time_values[0]
    for episode_index, trajectory in enumerate(trajectories):
        if len(trajectory) != time_values.size:
            raise ValueError("each trajectory must contain one set per time")
        sizes = np.asarray([len(set(item)) for item in trajectory], dtype=int)
        last_reduction_index = 0
        reductions_seen = 0
        for time_index in range(time_values.size - 1):
            if time_index > 0 and sizes[time_index] < sizes[time_index - 1]:
                last_reduction_index = time_index
                reductions_seen += 1
            future = np.flatnonzero(sizes[time_index + 1 :] < sizes[time_index])
            if future.size == 0:
                continue
            next_index = time_index + 1 + int(future[0])
            rows.append(
                [
                    (time_values[time_index] - time_values[0]) / span,
                    float(sizes[time_index]),
                    time_values[time_index] - time_values[last_reduction_index],
                    float(reductions_seen),
                ]
            )
            targets.append(float(time_values[next_index] - time_values[time_index]))
            episode_ids.append(episode_index)
            time_ids.append(time_index)
    if not rows:
        raise ValueError("no future uncertainty reductions exist in the supplied trajectories")
    return BifurcationTrainingData(
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(episode_ids, dtype=int),
        np.asarray(time_ids, dtype=int),
    )


class JackknifePlusRandomForestRegressor:
    """Random forest with explicit leave-one-out jackknife+ intervals."""

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 1,
        max_features: float | str | None = 1.0,
        random_state: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        if n_estimators < 1 or min_samples_leaf < 1:
            raise ValueError("n_estimators and min_samples_leaf must be positive")
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    def _make_model(self, seed: int):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("uncertainty-reduction forecasting requires the 'ml' extra") from exc
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JackknifePlusRandomForestRegressor":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
            raise ValueError("X/y must be a 2-D/1-D pair with equal sample count")
        if targets.size < 3 or not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("jackknife+ needs at least three finite samples")
        models = []
        residuals = np.empty(targets.size, dtype=float)
        for index in range(targets.size):
            keep = np.arange(targets.size) != index
            model = self._make_model(self.random_state + index + 1)
            model.fit(features[keep], targets[keep])
            residuals[index] = abs(targets[index] - float(model.predict(features[index : index + 1])[0]))
            models.append(model)
        self.models_ = tuple(models)
        self.residuals_ = residuals
        self.model_ = self._make_model(self.random_state).fit(features, targets)
        self.n_features_in_ = features.shape[1]
        return self

    def predict(
        self, X: np.ndarray, *, error_rate: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before predict")
        if not 0 < error_rate < 1:
            raise ValueError("error_rate must be in (0, 1)")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature shape")
        point = self.model_.predict(features)
        loo_predictions = np.vstack([model.predict(features) for model in self.models_])
        lower_candidates = loo_predictions - self.residuals_[:, None]
        upper_candidates = loo_predictions + self.residuals_[:, None]
        lower = np.quantile(lower_candidates, error_rate, axis=0, method="lower")
        upper = np.quantile(upper_candidates, 1.0 - error_rate, axis=0, method="higher")
        intervals = np.column_stack((np.maximum(0.0, lower), np.maximum(0.0, upper)))
        return np.maximum(0.0, point), intervals


class UncertaintyReductionForecaster:
    """Fit and query the random-forest jackknife+ bifurcation forecaster."""

    def __init__(self, **forest_parameters: object) -> None:
        self.forest_parameters = forest_parameters

    def fit(self, training_data: BifurcationTrainingData) -> "UncertaintyReductionForecaster":
        self.model_ = JackknifePlusRandomForestRegressor(**self.forest_parameters).fit(
            training_data.features, training_data.time_to_reduction
        )
        return self

    def forecast(
        self, features: Sequence[float], *, error_rate: float = 0.1
    ) -> BifurcationForecast:
        point, interval = self.model_.predict(np.asarray(features, dtype=float)[None, :], error_rate=error_rate)
        return BifurcationForecast(float(point[0]), float(interval[0, 0]), float(interval[0, 1]), error_rate)


class BifurcationDelayTimer:
    """Emit a forecast only after a stable run of consecutive predictions."""

    def __init__(self, delay: int = 3, tolerance: float = 1.0) -> None:
        if delay < 1 or tolerance < 0:
            raise ValueError("delay must be positive and tolerance nonnegative")
        self.delay = int(delay)
        self.tolerance = float(tolerance)
        self.reset()

    def reset(self) -> None:
        self._run: list[BifurcationForecast] = []

    def update(self, forecast: BifurcationForecast) -> BifurcationForecast | None:
        if self._run:
            previous = self._run[-1]
            intervals_overlap = forecast.lower <= previous.upper and previous.lower <= forecast.upper
            points_stable = abs(forecast.point - previous.point) <= self.tolerance
            if not (intervals_overlap and points_stable):
                self._run.clear()
        self._run.append(forecast)
        if len(self._run) < self.delay:
            return None
        recent = self._run[-self.delay :]
        return BifurcationForecast(
            point=float(np.median([item.point for item in recent])),
            lower=max(0.0, min(item.lower for item in recent)),
            upper=max(item.upper for item in recent),
            error_rate=max(item.error_rate for item in recent),
        )
