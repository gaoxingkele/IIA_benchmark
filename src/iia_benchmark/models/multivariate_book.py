"""Engineering implementations for the multivariate methods in Book Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import binom

from .univariate_book import BetaPosterior, beta_binomial_posterior


def _matrix(values: Iterable[Iterable[float]], *, minimum_rows: int = 3) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=float)
    if matrix.ndim != 2 or len(matrix) < minimum_rows or matrix.shape[1] < 2:
        raise ValueError("a finite two-dimensional sample matrix is required")
    if not np.isfinite(matrix).all():
        raise ValueError("samples contain non-finite values")
    return matrix


def _unit_directions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(values, axis=1)
    directions = np.zeros_like(values)
    nonzero = radius > 1e-12
    directions[nonzero] = values[nonzero] / radius[nonzero, None]
    return directions, radius


@dataclass
class SearchConeNOZAlarm:
    """Non-convex NOZ represented by angular search cones (Book Sec. 3.1.3)."""

    angular_resolution_degrees: float = 10.0
    radial_quantile: float = 1.0
    false_alarm_fraction: float = 0.0
    inference_batch_size: int = 2048

    def _keys(self, directions: np.ndarray) -> np.ndarray:
        if directions.shape[1] == 2:
            angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)[:, None]
            steps = np.deg2rad(self.angular_resolution_degrees)
            return np.floor(angles / steps).astype(int)
        angles = []
        for row in directions:
            coordinates = []
            for index in range(len(row) - 2):
                denominator = np.linalg.norm(row[index:])
                ratio = row[index] / denominator if denominator > 1e-12 else 1.0
                coordinates.append(np.arccos(np.clip(ratio, -1.0, 1.0)))
            coordinates.append(np.mod(np.arctan2(row[-1], row[-2]), 2.0 * np.pi))
            angles.append(coordinates)
        return np.floor(np.asarray(angles) / np.deg2rad(self.angular_resolution_degrees)).astype(int)

    def fit(self, normal_values: Iterable[Iterable[float]]) -> "SearchConeNOZAlarm":
        matrix = _matrix(normal_values)
        if not 0 < self.angular_resolution_degrees <= 90:
            raise ValueError("angular_resolution_degrees must be in (0, 90]")
        if not 0.5 <= self.radial_quantile <= 1.0:
            raise ValueError("radial_quantile must be in [0.5, 1]")
        if not 0 <= self.false_alarm_fraction < 0.5:
            raise ValueError("false_alarm_fraction must be in [0, 0.5)")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0, ddof=1)
        self.scale_[self.scale_ == 0] = 1.0
        normalized = (matrix - self.mean_) / self.scale_
        center = np.median(normalized, axis=0)
        distance = np.linalg.norm(normalized - center, axis=1)
        keep = max(matrix.shape[1] + 1, int(round(len(matrix) * (1 - self.false_alarm_fraction))))
        normalized = normalized[np.argsort(distance)[:keep]]
        directions, radii = _unit_directions(normalized)
        keys = self._keys(directions)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(tuple(key), []).append(index)
        self.cone_keys_ = list(groups)
        self.cone_index_ = {key: index for index, key in enumerate(self.cone_keys_)}
        self.cone_directions_ = np.asarray(
            [np.mean(directions[indices], axis=0) for indices in groups.values()]
        )
        cone_norm = np.linalg.norm(self.cone_directions_, axis=1)
        self.cone_directions_ /= np.maximum(cone_norm[:, None], 1e-12)
        self.cone_radii_ = np.asarray(
            [np.quantile(radii[indices], self.radial_quantile) for indices in groups.values()]
        )
        return self

    def decision_function(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "cone_radii_"):
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(list(values), dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean_):
            raise ValueError("values have the wrong shape")
        directions, radii = _unit_directions((matrix - self.mean_) / self.scale_)
        nearest = np.empty(len(directions), dtype=int)
        keys = self._keys(directions)
        unresolved = []
        for index, key in enumerate(keys):
            direct = self.cone_index_.get(tuple(key))
            if direct is None:
                unresolved.append(index)
            else:
                nearest[index] = direct
        for start in range(0, len(unresolved), self.inference_batch_size):
            indices = np.asarray(unresolved[start : start + self.inference_batch_size], dtype=int)
            if len(indices):
                similarity = directions[indices] @ self.cone_directions_.T
                nearest[indices] = np.argmax(similarity, axis=1)
        return radii - self.cone_radii_[nearest]

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        return (self.decision_function(values) > 1e-10).astype(np.int8)


def weighted_time_gradient(values: Sequence[float], forgetting_factor: float) -> float:
    """Weighted local-regression slope in Book equation (3.55)."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or not 0 < forgetting_factor <= 1:
        raise ValueError("a vector and forgetting_factor in (0, 1] are required")
    time = np.arange(len(samples), dtype=float)
    weights = forgetting_factor ** (len(samples) - 1 - time)
    weight_sum = float(np.sum(weights))
    denominator = float(np.sum(weights * time**2) * weight_sum - np.sum(weights * time) ** 2)
    if abs(denominator) < 1e-15:
        return 0.0
    numerator = float(np.sum(weights * time * samples) * weight_sum - np.sum(weights * samples) * np.sum(weights * time))
    return numerator / denominator


@dataclass
class AdaptiveTimeGradient:
    """Adaptive time-gradient extractor implementing equations (3.54)-(3.66)."""

    minimum_scale: int = 20
    maximum_scale: int = 100
    epsilon: float = 0.001

    def fit(self, training_values: Iterable[float]) -> "AdaptiveTimeGradient":
        samples = np.asarray(list(training_values), dtype=float)
        if self.minimum_scale < 2 or self.maximum_scale < self.minimum_scale:
            raise ValueError("invalid time-scale range")
        if len(samples) < self.maximum_scale:
            raise ValueError("training data shorter than maximum_scale")
        lambda_max = self.epsilon ** (1.0 / self.maximum_scale)
        gradients = [
            weighted_time_gradient(samples[end - self.maximum_scale : end], lambda_max)
            for end in range(self.maximum_scale, len(samples) + 1)
        ]
        self.significance_threshold_ = 2.0 * float(np.std(gradients, ddof=1))
        spans = [
            max(gradients[max(0, i - self.maximum_scale + 1) : i + 1])
            - min(gradients[max(0, i - self.maximum_scale + 1) : i + 1])
            for i in range(len(gradients))
        ]
        self.volatility_offset_ = float(np.quantile(spans, 0.1))
        volatility_range = max(float(np.quantile(spans, 0.95)) - self.volatility_offset_, 1e-12)
        lambda_min = self.epsilon ** (1.0 / self.minimum_scale)
        self.volatility_slope_ = (lambda_max - lambda_min) / volatility_range
        self.initial_forgetting_factor_ = lambda_max
        return self

    def transform(self, values: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "significance_threshold_"):
            raise RuntimeError("extractor is not fitted")
        samples = np.asarray(list(values), dtype=float)
        gradients = np.zeros(len(samples))
        scales = np.full(len(samples), self.maximum_scale, dtype=int)
        factor = self.initial_forgetting_factor_
        lambda_min = self.epsilon ** (1.0 / self.minimum_scale)
        lambda_max = self.epsilon ** (1.0 / self.maximum_scale)
        for index in range(1, len(samples)):
            start = max(0, index - self.maximum_scale + 1)
            gradients[index] = weighted_time_gradient(samples[start : index + 1], factor)
            scale = int(np.clip(round(np.log(self.epsilon) / np.log(factor)), self.minimum_scale, self.maximum_scale))
            scales[index] = scale
            recent = gradients[max(0, index - scale + 1) : index + 1]
            volatility = float(np.ptp(recent))
            factor = max(
                -self.volatility_slope_ * max(0.0, volatility - self.volatility_offset_) + lambda_max,
                lambda_min,
            )
        directions = np.where(
            gradients > self.significance_threshold_,
            1,
            np.where(gradients < -self.significance_threshold_, -1, 0),
        ).astype(np.int8)
        return gradients, directions, scales


@dataclass
class VariationDirectionAlarm:
    """Alarm when the ATG direction vector is outside the normal rule matrix."""

    minimum_scale: int = 20
    maximum_scale: int = 100

    def fit(self, normal_values: Iterable[Iterable[float]]) -> "VariationDirectionAlarm":
        matrix = _matrix(normal_values, minimum_rows=self.maximum_scale)
        self.extractors_ = []
        directions = []
        for column in matrix.T:
            extractor = AdaptiveTimeGradient(self.minimum_scale, self.maximum_scale).fit(column)
            self.extractors_.append(extractor)
            directions.append(extractor.transform(column)[1])
        direction_matrix = np.column_stack(directions)
        self.allowed_directions_ = {tuple(row) for row in direction_matrix[self.maximum_scale - 1 :]}
        return self

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "extractors_"):
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(list(values), dtype=float)
        directions = np.column_stack(
            [extractor.transform(matrix[:, index])[1] for index, extractor in enumerate(self.extractors_)]
        )
        return np.asarray([tuple(row) not in self.allowed_directions_ for row in directions], dtype=np.int8)


@dataclass
class BayesianWindowRegressionAlarm:
    """Bayesian/ridge time-varying regression with abnormal-window update freeze."""

    window_length: int | None = None
    forgetting_factor: float = 0.95
    ridge: float = 1e-6
    target_false_alarm_rate: float = 0.05
    significance: float = 0.05
    threshold_quantile: float = 0.95

    def _design(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(values)), values])

    def fit(self, predictors: Iterable[Iterable[float]], response: Iterable[float]) -> "BayesianWindowRegressionAlarm":
        x = _matrix(predictors)
        y = np.asarray(list(response), dtype=float)
        if len(x) != len(y):
            raise ValueError("predictors and response lengths differ")
        self.window_length_ = self.window_length or x.shape[1] + 2
        design = self._design(x)
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self.mean_ = np.linalg.solve(gram, design.T @ y)
        residual = y - design @ self.mean_
        self.noise_variance_ = max(float(np.var(residual, ddof=design.shape[1])), 1e-12)
        self.covariance_ = self.noise_variance_ * np.linalg.pinv(gram)
        cumulative = np.convolve(residual, np.ones(self.window_length_), mode="valid")
        self.alarm_threshold_ = float(np.quantile(np.abs(cumulative), self.threshold_quantile))
        return self

    def predict_update(self, predictors: Iterable[Iterable[float]], response: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "mean_"):
            raise RuntimeError("model is not fitted")
        x = np.asarray(list(predictors), dtype=float)
        y = np.asarray(list(response), dtype=float)
        design = self._design(x)
        alarms = np.zeros(len(x), dtype=np.int8)
        frozen = np.zeros(len(x), dtype=np.int8)
        residual_window: list[float] = []
        alarm_window: list[int] = []
        threshold_count = int(binom.ppf(1.0 - self.significance, self.window_length_, self.target_false_alarm_rate))
        for index, (row, observed) in enumerate(zip(design, y, strict=True)):
            residual_window.append(float(observed - row @ self.mean_))
            residual_window = residual_window[-self.window_length_ :]
            alarms[index] = int(abs(sum(residual_window)) > self.alarm_threshold_)
            alarm_window.append(int(alarms[index]))
            if (index + 1) % self.window_length_ == 0:
                window_x = design[index + 1 - self.window_length_ : index + 1]
                window_y = y[index + 1 - self.window_length_ : index + 1]
                if sum(alarm_window[-self.window_length_ :]) > threshold_count:
                    frozen[index + 1 - self.window_length_ : index + 1] = 1
                    continue
                prior_precision = np.linalg.pinv(self.covariance_)
                precision = (window_x.T @ window_x + self.ridge * np.eye(window_x.shape[1])) / self.noise_variance_ + self.forgetting_factor * prior_precision
                rhs = window_x.T @ window_y / self.noise_variance_ + self.forgetting_factor * prior_precision @ self.mean_
                self.covariance_ = np.linalg.pinv(precision)
                self.mean_ = self.covariance_ @ rhs
        return alarms, frozen


@dataclass(frozen=True)
class CondenserParameters:
    latent_heat: float
    tube_count: float
    tube_length: float
    steam_load: float
    inner_diameter: float
    outer_diameter: float


@dataclass(frozen=True)
class CondenserAlarmRateBounds:
    """Bayesian FAR/MAR intervals in Book equations (3.119) and (3.127)."""

    false_alarm: BetaPosterior
    missed_alarm: BetaPosterior


def condenser_alarm_rate_bounds(
    normal_alarm: Iterable[int | bool],
    abnormal_alarm: Iterable[int | bool],
    *,
    confidence: float = 0.99,
) -> CondenserAlarmRateBounds:
    normal = np.asarray(list(normal_alarm), dtype=bool)
    abnormal = np.asarray(list(abnormal_alarm), dtype=bool)
    if normal.ndim != 1 or abnormal.ndim != 1 or not len(normal) or not len(abnormal):
        raise ValueError("normal_alarm and abnormal_alarm must be non-empty vectors")
    return CondenserAlarmRateBounds(
        beta_binomial_posterior(int(np.sum(normal)), len(normal), confidence=confidence),
        beta_binomial_posterior(int(np.sum(~abnormal)), len(abnormal), confidence=confidence),
    )


@dataclass
class CondenserPhysicalModel:
    """Book equations (3.90)-(3.103) for condenser pressure."""

    water_density: float = 1000.0
    water_heat_capacity: float = 4.1816

    @staticmethod
    def saturation_pressure(steam_temperature: np.ndarray | float) -> np.ndarray:
        """Return saturation pressure in kPa for Book equation (3.90)."""

        temperature = np.asarray(steam_temperature, dtype=float)
        return 9.8e-3 * ((temperature + 100.0) / 57.66) ** 7.46

    def predict_pressure(
        self,
        steam_flow: Iterable[float],
        inlet_temperature: Iterable[float],
        outlet_temperature: Iterable[float],
        parameters: CondenserParameters,
    ) -> np.ndarray:
        dc = np.asarray(list(steam_flow), dtype=float)
        t1 = np.asarray(list(inlet_temperature), dtype=float)
        t2 = np.asarray(list(outlet_temperature), dtype=float)
        difference = np.maximum(t2 - t1, 1e-6)
        cooling_flow = parameters.latent_heat * dc / (self.water_heat_capacity * difference)
        area = 2.0 * np.pi * parameters.outer_diameter * parameters.tube_length * parameters.tube_count
        omega = np.where(t1 <= 26.7, 0.0969 * (1.0 + 0.15 * t1), 0.4845)
        velocity_term = 8.8 * cooling_flow / (
            self.water_density * np.pi * parameters.tube_count * parameters.inner_diameter ** 2.25
        )
        eta_w = np.maximum(velocity_term, 1e-12) ** omega
        eta_t = np.where(
            t1 <= 35.0,
            1.0 - (0.52 - 0.0072 * parameters.steam_load) * (35.0 - t1) / 1000.0,
            1.0 + 0.002 * (t1 - 35.0),
        )
        transfer = 3.2865 * eta_w * eta_t
        terminal_difference = difference / np.expm1(
            np.maximum(transfer * area / (self.water_heat_capacity * cooling_flow), 1e-12)
        )
        return self.saturation_pressure(t2 + terminal_difference)

    @staticmethod
    def goodness_of_fit(observed: Iterable[float], predicted: Iterable[float]) -> float:
        observed_array = np.asarray(list(observed), dtype=float)
        predicted_array = np.asarray(list(predicted), dtype=float)
        denominator = float(np.sum((observed_array - np.mean(observed_array)) ** 2))
        return 1.0 - float(np.sum((observed_array - predicted_array) ** 2)) / denominator if denominator else 0.0

    def fit(
        self,
        pressure: Iterable[float],
        steam_flow: Iterable[float],
        inlet_temperature: Iterable[float],
        outlet_temperature: Iterable[float],
        *,
        bounds: Sequence[tuple[float, float]],
        seed: int = 0,
        maxiter: int = 100,
    ) -> "CondenserPhysicalModel":
        observed = np.asarray(list(pressure), dtype=float)
        dc, t1, t2 = list(steam_flow), list(inlet_temperature), list(outlet_temperature)

        def objective(vector: np.ndarray) -> float:
            params = CondenserParameters(*map(float, vector))
            predicted = self.predict_pressure(dc, t1, t2, params)
            return -self.goodness_of_fit(observed, predicted)

        result = differential_evolution(objective, bounds=bounds, seed=seed, maxiter=maxiter, polish=True)
        self.parameters_ = CondenserParameters(*map(float, result.x))
        self.fit_score_ = float(-result.fun)
        return self


@dataclass
class CondenserNOZAlarm:
    """Physics-residual plus search-cone operating-zone monitor."""

    parameters: CondenserParameters
    angular_resolution_degrees: float = 10.0
    residual_quantile: float = 0.99

    def fit(self, normal_values: Iterable[Iterable[float]]) -> "CondenserNOZAlarm":
        matrix = _matrix(normal_values)
        if matrix.shape[1] != 4:
            raise ValueError("columns must be [pressure, steam_flow, inlet_temperature, outlet_temperature]")
        physical = CondenserPhysicalModel()
        predicted = physical.predict_pressure(matrix[:, 1], matrix[:, 2], matrix[:, 3], self.parameters)
        self.residual_threshold_ = float(np.quantile(np.abs(matrix[:, 0] - predicted), self.residual_quantile))
        self.zone_ = SearchConeNOZAlarm(self.angular_resolution_degrees).fit(matrix)
        return self

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        matrix = np.asarray(list(values), dtype=float)
        physical = CondenserPhysicalModel()
        predicted = physical.predict_pressure(matrix[:, 1], matrix[:, 2], matrix[:, 3], self.parameters)
        physical_alarm = np.abs(matrix[:, 0] - predicted) > self.residual_threshold_
        return np.maximum(physical_alarm.astype(np.int8), self.zone_.predict(matrix))
