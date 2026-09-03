"""Leakage-safe adapters for transferring univariate alarm methods.

The adapters normalize or select a signal before it reaches a book method.
They do not alter the equations in :mod:`iia_benchmark.models.univariate_book`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import ks_2samp, rankdata

from .univariate_book import AlarmOnOffDelay


Tail = Literal["high", "low", "two_sided"]


def _one_dimensional(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) < 2:
        raise ValueError(f"{name} must contain at least two samples and two features")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _auc(normal_score: np.ndarray, abnormal_score: np.ndarray) -> float:
    combined = np.r_[normal_score, abnormal_score]
    ranks = rankdata(combined, method="average")
    n0 = len(normal_score)
    n1 = len(abnormal_score)
    statistic = float(np.sum(ranks[n0:]) - n1 * (n1 + 1) / 2.0)
    return statistic / (n0 * n1)


def _lag_one(values: np.ndarray) -> float:
    if np.std(values[:-1]) <= 1e-12 or np.std(values[1:]) <= 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


class RobustMedianScaler:
    """Median/MAD scaler with deterministic IQR and standard-deviation fallbacks."""

    def fit(self, values: Sequence[float] | np.ndarray) -> "RobustMedianScaler":
        samples = _one_dimensional(values, "values")
        self.location_ = float(np.median(samples))
        mad = float(np.median(np.abs(samples - self.location_)))
        scale = 1.4826 * mad
        self.scale_source_ = "mad"
        if scale <= 1e-12:
            iqr = float(np.quantile(samples, 0.75) - np.quantile(samples, 0.25))
            scale = iqr / 1.349
            self.scale_source_ = "iqr"
        if scale <= 1e-12:
            scale = float(np.std(samples, ddof=1))
            self.scale_source_ = "standard_deviation"
        if scale <= 1e-12:
            raise ValueError("values are constant")
        self.scale_ = scale
        return self

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("RobustMedianScaler is not fitted")
        samples = _one_dimensional(values, "values")
        return (samples - self.location_) / self.scale_

    def inverse_transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("RobustMedianScaler is not fitted")
        samples = _one_dimensional(values, "values")
        return samples * self.scale_ + self.location_

    def fit_transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


class EmpiricalCDFNormalizer:
    """Map a signal to finite-sample empirical probabilities from normal data."""

    def fit(self, normal: Sequence[float] | np.ndarray) -> "EmpiricalCDFNormalizer":
        samples = _one_dimensional(normal, "normal")
        if float(np.ptp(samples)) <= 1e-12:
            raise ValueError("normal reference is constant")
        self.reference_ = np.sort(samples)
        return self

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("EmpiricalCDFNormalizer is not fitted")
        samples = _one_dimensional(values, "values")
        left = np.searchsorted(self.reference_, samples, side="left")
        right = np.searchsorted(self.reference_, samples, side="right")
        midrank = (left + right + 1.0) / 2.0
        probabilities = midrank / (len(self.reference_) + 1.0)
        epsilon = 0.5 / (len(self.reference_) + 1.0)
        return np.clip(probabilities, epsilon, 1.0 - epsilon)

    def fit_transform(self, normal: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.fit(normal).transform(normal)


class EmpiricalCDFAlarm:
    """Distribution-free tail alarm calibrated from target-domain normal data."""

    def __init__(
        self,
        *,
        tail_probability: float = 0.05,
        tail: Tail = "high",
        delay: int = 1,
    ) -> None:
        if not 0.0 < tail_probability < 0.5:
            raise ValueError("tail_probability must be between zero and 0.5")
        if tail not in ("high", "low", "two_sided"):
            raise ValueError("tail must be high, low, or two_sided")
        if delay < 1:
            raise ValueError("delay must be at least one")
        self.tail_probability = float(tail_probability)
        self.tail = tail
        self.delay = int(delay)

    def fit(self, normal: Sequence[float] | np.ndarray) -> "EmpiricalCDFAlarm":
        self.normalizer_ = EmpiricalCDFNormalizer().fit(normal)
        self.threshold_ = 1.0 - self.tail_probability
        return self

    def score_samples(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "normalizer_"):
            raise RuntimeError("EmpiricalCDFAlarm is not fitted")
        probabilities = self.normalizer_.transform(values)
        if self.tail == "high":
            return probabilities
        if self.tail == "low":
            return 1.0 - probabilities
        return 2.0 * np.abs(probabilities - 0.5)

    def predict(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        score = self.score_samples(values)
        return AlarmOnOffDelay(self.threshold_, "high", self.delay).predict(score)


@dataclass(frozen=True)
class FeatureStabilityDiagnostic:
    index: int
    name: str
    direction: Literal["high", "low"]
    score: float
    minimum_block_separation_sd: float
    minimum_block_auc: float
    direction_consistency: float
    internal_normal_ks: float
    absolute_lag_one: float
    valid: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FeatureStabilitySelector:
    """Select a feature for block-stable separation, not peak calibration shift."""

    def __init__(
        self,
        *,
        chronological_blocks: int = 3,
        normal_drift_weight: float = 1.0,
        direction_instability_weight: float = 2.0,
        temporal_dependence_weight: float = 0.25,
    ) -> None:
        if chronological_blocks < 2:
            raise ValueError("chronological_blocks must be at least two")
        for value in (
            normal_drift_weight,
            direction_instability_weight,
            temporal_dependence_weight,
        ):
            if value < 0:
                raise ValueError("selector weights must be non-negative")
        self.chronological_blocks = int(chronological_blocks)
        self.normal_drift_weight = float(normal_drift_weight)
        self.direction_instability_weight = float(direction_instability_weight)
        self.temporal_dependence_weight = float(temporal_dependence_weight)

    def fit(
        self,
        normal: Sequence[Sequence[float]] | np.ndarray,
        abnormal: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "FeatureStabilitySelector":
        normal_values = _matrix(normal, "normal")
        abnormal_values = _matrix(abnormal, "abnormal")
        if normal_values.shape[1] != abnormal_values.shape[1]:
            raise ValueError("normal and abnormal feature counts differ")
        names = (
            tuple(str(value) for value in feature_names)
            if feature_names is not None
            else tuple(f"feature_{index}" for index in range(normal_values.shape[1]))
        )
        if len(names) != normal_values.shape[1]:
            raise ValueError("feature_names length differs from feature count")
        normal_split = len(normal_values) // 2
        abnormal_blocks = np.array_split(abnormal_values, self.chronological_blocks)
        if normal_split < 2 or len(normal_values) - normal_split < 2:
            raise ValueError("normal data are too short for chronological validation")
        if any(len(block) < 2 for block in abnormal_blocks):
            raise ValueError("abnormal data are too short for chronological blocks")

        diagnostics: list[FeatureStabilityDiagnostic] = []
        for index, name in enumerate(names):
            reference = normal_values[:, index]
            scale = float(np.std(reference, ddof=1))
            if scale <= 1e-12:
                diagnostics.append(
                    FeatureStabilityDiagnostic(
                        index=index,
                        name=name,
                        direction="high",
                        score=float("-inf"),
                        minimum_block_separation_sd=float("-inf"),
                        minimum_block_auc=0.5,
                        direction_consistency=0.0,
                        internal_normal_ks=0.0,
                        absolute_lag_one=0.0,
                        valid=False,
                    )
                )
                continue
            raw_delta = float(
                np.median(abnormal_values[:, index]) - np.median(reference)
            )
            direction: Literal["high", "low"] = "high" if raw_delta >= 0 else "low"
            sign = 1.0 if direction == "high" else -1.0
            separations = np.asarray(
                [
                    sign
                    * (np.median(block[:, index]) - np.median(reference))
                    / scale
                    for block in abnormal_blocks
                ],
                dtype=float,
            )
            aucs = np.asarray(
                [_auc(sign * reference, sign * block[:, index]) for block in abnormal_blocks]
            )
            consistency = float(np.mean(separations > 1e-12))
            normal_ks = float(
                ks_2samp(
                    normal_values[:normal_split, index],
                    normal_values[normal_split:, index],
                ).statistic
            )
            autocorrelation = abs(_lag_one(reference))
            score = float(
                np.min(separations)
                - self.normal_drift_weight * normal_ks
                - self.direction_instability_weight * (1.0 - consistency)
                - self.temporal_dependence_weight * max(0.0, autocorrelation - 0.5)
            )
            diagnostics.append(
                FeatureStabilityDiagnostic(
                    index=index,
                    name=name,
                    direction=direction,
                    score=score,
                    minimum_block_separation_sd=float(np.min(separations)),
                    minimum_block_auc=float(np.min(aucs)),
                    direction_consistency=consistency,
                    internal_normal_ks=normal_ks,
                    absolute_lag_one=autocorrelation,
                    valid=True,
                )
            )
        valid = [row for row in diagnostics if row.valid]
        if not valid:
            raise ValueError("no nonconstant feature is available")
        selected = max(valid, key=lambda row: (row.score, row.minimum_block_auc, -row.index))
        self.feature_index_ = selected.index
        self.feature_name_ = selected.name
        self.direction_ = selected.direction
        self.diagnostics_ = tuple(diagnostics)
        return self

    def transform(self, values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "feature_index_"):
            raise RuntimeError("FeatureStabilitySelector is not fitted")
        matrix = _matrix(values, "values")
        if self.feature_index_ >= matrix.shape[1]:
            raise ValueError("values do not contain the selected feature")
        return matrix[:, self.feature_index_]
