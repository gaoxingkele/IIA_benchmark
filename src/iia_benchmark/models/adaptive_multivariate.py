"""Distribution-aware adapters for multivariate industrial alarm transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from ..evaluation.multivariate_distribution_audit import (
    MultivariateApplicability,
    MultivariateApplicabilityThresholds,
    assess_multivariate_calibration,
)
from .univariate_book import AlarmOnOffDelay


def _matrix(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 2 or result.shape[0] < 4 or result.shape[1] < 2:
        raise ValueError(f"{name} must contain at least four rows and two features")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


class RobustShrinkageMahalanobisAlarm:
    """Mahalanobis alarm with median/MAD scaling and ridge covariance shrinkage."""

    def __init__(self, *, quantile: float = 0.99, shrinkage: float = 0.10) -> None:
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and one")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between zero and one")
        self.quantile = float(quantile)
        self.shrinkage = float(shrinkage)

    def fit(
        self, normal_values: Iterable[Iterable[float]]
    ) -> "RobustShrinkageMahalanobisAlarm":
        matrix = _matrix(normal_values, "normal_values")
        self.location_ = np.median(matrix, axis=0)
        mad = 1.4826 * np.median(np.abs(matrix - self.location_), axis=0)
        iqr = (
            np.quantile(matrix, 0.75, axis=0)
            - np.quantile(matrix, 0.25, axis=0)
        ) / 1.349
        standard = np.std(matrix, axis=0, ddof=1)
        self.scale_ = np.where(mad > 1e-12, mad, np.where(iqr > 1e-12, iqr, standard))
        self.constant_features_ = tuple(
            int(index) for index in np.flatnonzero(self.scale_ <= 1e-12)
        )
        self.scale_[self.scale_ <= 1e-12] = 1.0
        normalized = (matrix - self.location_) / self.scale_
        covariance = np.atleast_2d(np.cov(normalized, rowvar=False))
        target_variance = max(float(np.trace(covariance) / len(covariance)), 1e-12)
        shrunk = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * target_variance * np.eye(len(covariance))
        )
        self.covariance_ = shrunk
        self.precision_ = np.linalg.pinv(shrunk)
        scores = self.score_samples(matrix)
        self.threshold_ = float(np.quantile(scores, self.quantile))
        self.covariance_condition_ = float(np.linalg.cond(shrunk))
        return self

    def score_samples(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "precision_"):
            raise RuntimeError("RobustShrinkageMahalanobisAlarm is not fitted")
        matrix = np.asarray(list(values), dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.location_):
            raise ValueError("values have the wrong feature shape")
        if not np.isfinite(matrix).all():
            raise ValueError("values contain non-finite values")
        centered = (matrix - self.location_) / self.scale_
        return np.einsum("ij,jk,ik->i", centered, self.precision_, centered)

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        return (self.score_samples(values) > self.threshold_).astype(np.int8)


@dataclass(frozen=True)
class MultivariateBlockCalibrationCandidate:
    reference_window: int
    delay: int
    threshold: float
    point_false_alarm_rate: float
    block_alarm_rate: float
    loss: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BlockCalibratedRobustMahalanobisAlarm:
    """Chronologically calibrate robust multivariate scores on normal blocks."""

    def __init__(
        self,
        *,
        tail_probability: float = 0.05,
        shrinkage: float = 0.10,
        reference_windows: Sequence[int] = (128, 256, 512, 1024),
        delays: Sequence[int] = (1, 2, 3, 4, 5),
        validation_fraction: float = 0.30,
        block_size: int = 60,
        target_point_false_alarm_rate: float = 0.05,
        target_block_alarm_rate: float = 0.20,
        block_weight: float = 0.25,
    ) -> None:
        if not 0.0 < tail_probability < 0.5:
            raise ValueError("tail_probability must be between zero and 0.5")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between zero and one")
        if not reference_windows or any(int(value) < 16 for value in reference_windows):
            raise ValueError("reference_windows must contain values of at least 16")
        if not delays or any(int(value) < 1 for value in delays):
            raise ValueError("delays must be positive")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5")
        if block_size < 2:
            raise ValueError("block_size must be at least two")
        self.tail_probability = float(tail_probability)
        self.shrinkage = float(shrinkage)
        self.reference_windows = tuple(sorted({int(value) for value in reference_windows}))
        self.delays = tuple(sorted({int(value) for value in delays}))
        self.validation_fraction = float(validation_fraction)
        self.block_size = int(block_size)
        self.target_point_false_alarm_rate = float(target_point_false_alarm_rate)
        self.target_block_alarm_rate = float(target_block_alarm_rate)
        self.block_weight = float(block_weight)

    def _block_alarm_rate(self, alarm: np.ndarray) -> float:
        blocks = [
            alarm[start : start + self.block_size]
            for start in range(0, len(alarm), self.block_size)
        ]
        return float(np.mean([bool(np.any(block)) for block in blocks]))

    def fit(
        self, normal_values: Iterable[Iterable[float]]
    ) -> "BlockCalibratedRobustMahalanobisAlarm":
        matrix = _matrix(normal_values, "normal_values")
        split = int(np.floor(len(matrix) * (1.0 - self.validation_fraction)))
        split = min(max(split, matrix.shape[1] + 3), len(matrix) - 4)
        reference_pool = matrix[:split]
        validation = matrix[split:]
        windows = tuple(value for value in self.reference_windows if value <= len(reference_pool))
        if not windows:
            windows = (len(reference_pool),)
        candidates: list[MultivariateBlockCalibrationCandidate] = []
        models: dict[int, RobustShrinkageMahalanobisAlarm] = {}
        thresholds: dict[int, float] = {}
        for window in windows:
            model = RobustShrinkageMahalanobisAlarm(
                quantile=1.0 - self.tail_probability,
                shrinkage=self.shrinkage,
            ).fit(reference_pool[-window:])
            scores = model.score_samples(validation)
            threshold = float(np.quantile(scores, 1.0 - self.tail_probability))
            models[window] = model
            thresholds[window] = threshold
            for delay in self.delays:
                alarm = AlarmOnOffDelay(threshold, "high", delay).predict(scores)
                point_rate = float(np.mean(alarm))
                block_rate = self._block_alarm_rate(alarm)
                loss = (
                    abs(point_rate - self.target_point_false_alarm_rate)
                    + self.block_weight
                    * abs(block_rate - self.target_block_alarm_rate)
                )
                candidates.append(
                    MultivariateBlockCalibrationCandidate(
                        reference_window=window,
                        delay=delay,
                        threshold=threshold,
                        point_false_alarm_rate=point_rate,
                        block_alarm_rate=block_rate,
                        loss=float(loss),
                    )
                )
        selected = min(
            candidates,
            key=lambda row: (row.loss, row.delay, -row.reference_window),
        )
        self.model_ = models[selected.reference_window]
        self.threshold_ = thresholds[selected.reference_window]
        self.selected_reference_window_ = selected.reference_window
        self.selected_delay_ = selected.delay
        self.calibration_candidates_ = tuple(candidates)
        return self

    def score_samples(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("BlockCalibratedRobustMahalanobisAlarm is not fitted")
        return self.model_.score_samples(values)

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        scores = self.score_samples(values)
        return AlarmOnOffDelay(
            self.threshold_, "high", self.selected_delay_
        ).predict(scores)


@dataclass(frozen=True)
class AdaptiveMultivariateDecision:
    status: Literal["static", "adapt", "reject_multivariate"]
    selected_model: str | None
    applicability: MultivariateApplicability
    reason: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["applicability"] = self.applicability.as_dict()
        return result


class AdaptiveMultivariateAlarmRouter:
    """Choose static, block-calibrated, or denied multivariate scoring."""

    def __init__(
        self,
        *,
        applicability_thresholds: MultivariateApplicabilityThresholds | None = None,
        tail_probability: float = 0.05,
        shrinkage: float = 0.10,
        reference_windows: Sequence[int] = (128, 256, 512, 1024),
        delays: Sequence[int] = (1, 2, 3, 4, 5),
        block_size: int = 60,
    ) -> None:
        self.applicability_thresholds = (
            applicability_thresholds or MultivariateApplicabilityThresholds()
        )
        self.tail_probability = float(tail_probability)
        self.shrinkage = float(shrinkage)
        self.reference_windows = tuple(int(value) for value in reference_windows)
        self.delays = tuple(int(value) for value in delays)
        self.block_size = int(block_size)

    def fit(
        self,
        normal_values: Iterable[Iterable[float]],
        abnormal_calibration: Iterable[Iterable[float]],
    ) -> "AdaptiveMultivariateAlarmRouter":
        normal = _matrix(normal_values, "normal_values")
        abnormal = _matrix(abnormal_calibration, "abnormal_calibration")
        if normal.shape[1] != abnormal.shape[1]:
            raise ValueError("normal and abnormal feature counts differ")
        applicability = assess_multivariate_calibration(
            normal,
            abnormal,
            thresholds=self.applicability_thresholds,
        )
        self.model_ = None
        if applicability.status == "reject_multivariate":
            selected = None
            reason = "calibration blocks do not support a stable multivariate score"
        elif applicability.status == "adapt":
            self.model_ = BlockCalibratedRobustMahalanobisAlarm(
                tail_probability=self.tail_probability,
                shrinkage=self.shrinkage,
                reference_windows=self.reference_windows,
                delays=self.delays,
                block_size=self.block_size,
            ).fit(normal)
            selected = "block_calibrated_robust_shrinkage_mahalanobis"
            reason = "normal distribution or temporal shift requires block calibration"
        else:
            self.model_ = RobustShrinkageMahalanobisAlarm(
                quantile=1.0 - self.tail_probability,
                shrinkage=self.shrinkage,
            ).fit(normal)
            selected = "robust_shrinkage_mahalanobis"
            reason = "calibration distribution supports a static robust score"
        self.decision_ = AdaptiveMultivariateDecision(
            status=applicability.status,
            selected_model=selected,
            applicability=applicability,
            reason=reason,
        )
        return self

    def predict(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        if not hasattr(self, "decision_"):
            raise RuntimeError("AdaptiveMultivariateAlarmRouter is not fitted")
        if self.model_ is None:
            raise RuntimeError("multivariate prediction denied by calibration gate")
        return self.model_.predict(values)
