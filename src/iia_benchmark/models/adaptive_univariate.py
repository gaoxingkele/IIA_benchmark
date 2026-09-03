"""Leakage-safe adapters for transferring univariate alarm methods.

The adapters normalize or select a signal before it reaches a book method.
They do not alter the equations in :mod:`iia_benchmark.models.univariate_book`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import ks_2samp, rankdata

from ..evaluation.distribution_audit import (
    ApplicabilityThresholds,
    CalibrationApplicability,
    assess_univariate_calibration,
)
from .univariate import ThresholdDelayDeadband
from .univariate_book import (
    AlarmOnOffDelay,
    alarm_episode_metrics,
    deadband_index,
    design_deadband_width,
    design_iid_delay_timer,
    design_non_iid_delay_timer,
    select_alarm_probability_threshold,
)


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
class BlockCalibrationCandidate:
    reference_window: int
    delay: int
    point_false_alarm_rate: float
    block_alarm_rate: float
    loss: float


class BlockCalibratedECDFAlarm:
    """Select a recent reference window and delay on chronological normal data."""

    def __init__(
        self,
        *,
        tail_probability: float = 0.05,
        tail: Tail = "high",
        reference_windows: Sequence[int] = (128, 256, 512, 1024),
        delays: Sequence[int] = (1, 2, 3, 4, 5),
        validation_fraction: float = 0.30,
        block_size: int = 60,
        target_point_false_alarm_rate: float = 0.05,
        target_block_alarm_rate: float = 0.20,
        block_weight: float = 0.25,
    ) -> None:
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5")
        if block_size < 2:
            raise ValueError("block_size must be at least two")
        if not reference_windows or any(int(value) < 16 for value in reference_windows):
            raise ValueError("reference_windows must contain values of at least 16")
        if not delays or any(int(value) < 1 for value in delays):
            raise ValueError("delays must be positive")
        if not 0.0 < target_point_false_alarm_rate < 1.0:
            raise ValueError("target_point_false_alarm_rate must be between zero and one")
        if not 0.0 < target_block_alarm_rate < 1.0:
            raise ValueError("target_block_alarm_rate must be between zero and one")
        if block_weight < 0:
            raise ValueError("block_weight must be non-negative")
        self.tail_probability = float(tail_probability)
        self.tail = tail
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
        return float(np.mean([np.any(block) for block in blocks]))

    def fit(self, normal: Sequence[float] | np.ndarray) -> "BlockCalibratedECDFAlarm":
        samples = _one_dimensional(normal, "normal")
        boundary = int(np.floor(len(samples) * (1.0 - self.validation_fraction)))
        if boundary < 32 or len(samples) - boundary < 8:
            raise ValueError("normal reference is too short for chronological calibration")
        fit_values = samples[:boundary]
        validation = samples[boundary:]
        windows = [value for value in self.reference_windows if value <= len(fit_values)]
        if not windows:
            windows = [len(fit_values)]
        candidates: list[BlockCalibrationCandidate] = []
        for window in windows:
            for delay in self.delays:
                model = EmpiricalCDFAlarm(
                    tail_probability=self.tail_probability,
                    tail=self.tail,
                    delay=delay,
                ).fit(fit_values[-window:])
                alarm = model.predict(validation)
                point_far = float(np.mean(alarm))
                block_far = self._block_alarm_rate(alarm)
                loss = float(
                    abs(point_far - self.target_point_false_alarm_rate)
                    / self.target_point_false_alarm_rate
                    + self.block_weight
                    * abs(block_far - self.target_block_alarm_rate)
                    / self.target_block_alarm_rate
                )
                candidates.append(
                    BlockCalibrationCandidate(
                        reference_window=window,
                        delay=delay,
                        point_false_alarm_rate=point_far,
                        block_alarm_rate=block_far,
                        loss=loss,
                    )
                )
        selected = min(
            candidates,
            key=lambda row: (
                row.loss,
                abs(row.point_false_alarm_rate - self.target_point_false_alarm_rate),
                -row.reference_window,
                row.delay,
            ),
        )
        self.selected_reference_window_ = selected.reference_window
        self.selected_delay_ = selected.delay
        self.calibration_candidates_ = tuple(candidates)
        self.model_ = EmpiricalCDFAlarm(
            tail_probability=self.tail_probability,
            tail=self.tail,
            delay=self.selected_delay_,
        ).fit(samples[-self.selected_reference_window_ :])
        return self

    def score_samples(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("BlockCalibratedECDFAlarm is not fitted")
        return self.model_.score_samples(values)

    def predict(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("BlockCalibratedECDFAlarm is not fitted")
        return self.model_.predict(values)


class RegimeConditionalECDFAlarm:
    """Maintain a separate empirical normal distribution for each known regime."""

    def __init__(
        self,
        *,
        tail_probability: float = 0.05,
        tail: Tail = "two_sided",
        delay: int = 1,
    ) -> None:
        self.tail_probability = float(tail_probability)
        self.tail = tail
        self.delay = int(delay)

    def fit(
        self,
        normal: Sequence[float] | np.ndarray,
        regimes: Sequence[object] | np.ndarray,
    ) -> "RegimeConditionalECDFAlarm":
        samples = _one_dimensional(normal, "normal")
        labels = np.asarray(regimes)
        if labels.ndim != 1 or len(labels) != len(samples):
            raise ValueError("regimes must be one-dimensional and match normal")
        self.models_ = {}
        for label in np.unique(labels):
            mask = labels == label
            if int(np.sum(mask)) < 16:
                raise ValueError(f"regime {label!r} has fewer than 16 normal samples")
            self.models_[label.item() if hasattr(label, "item") else label] = EmpiricalCDFAlarm(
                tail_probability=self.tail_probability,
                tail=self.tail,
                delay=1,
            ).fit(samples[mask])
        return self

    def score_samples(
        self,
        values: Sequence[float] | np.ndarray,
        regimes: Sequence[object] | np.ndarray,
    ) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("RegimeConditionalECDFAlarm is not fitted")
        samples = _one_dimensional(values, "values")
        labels = np.asarray(regimes)
        if labels.ndim != 1 or len(labels) != len(samples):
            raise ValueError("regimes must be one-dimensional and match values")
        result = np.empty(len(samples), dtype=float)
        for label in np.unique(labels):
            key = label.item() if hasattr(label, "item") else label
            if key not in self.models_:
                raise ValueError(f"unseen regime {key!r}")
            mask = labels == label
            result[mask] = self.models_[key].score_samples(samples[mask])
        return result

    def predict(
        self,
        values: Sequence[float] | np.ndarray,
        regimes: Sequence[object] | np.ndarray,
    ) -> np.ndarray:
        scores = self.score_samples(values, regimes)
        threshold = 1.0 - self.tail_probability
        return AlarmOnOffDelay(threshold, "high", self.delay).predict(scores)


class SafeRollingECDFAlarm:
    """Update an empirical reference only with central, non-suspect observations."""

    def __init__(
        self,
        *,
        tail_probability: float = 0.05,
        tail: Tail = "two_sided",
        delay: int = 1,
        reference_window: int = 512,
        update_guard_score: float = 0.80,
    ) -> None:
        if reference_window < 32:
            raise ValueError("reference_window must be at least 32")
        if not 0.5 < update_guard_score < 1.0 - tail_probability:
            raise ValueError("update_guard_score must be between 0.5 and the alarm threshold")
        self.tail_probability = float(tail_probability)
        self.tail = tail
        self.delay = int(delay)
        self.reference_window = int(reference_window)
        self.update_guard_score = float(update_guard_score)

    def fit(self, normal: Sequence[float] | np.ndarray) -> "SafeRollingECDFAlarm":
        samples = _one_dimensional(normal, "normal")
        if len(samples) < self.reference_window:
            raise ValueError("normal contains fewer samples than reference_window")
        self.initial_reference_ = samples[-self.reference_window :].copy()
        EmpiricalCDFNormalizer().fit(self.initial_reference_)
        return self

    def score_samples(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "initial_reference_"):
            raise RuntimeError("SafeRollingECDFAlarm is not fitted")
        samples = _one_dimensional(values, "values")
        reference = list(self.initial_reference_)
        scores = np.empty(len(samples), dtype=float)
        updates = 0
        frozen = 0
        for index, value in enumerate(samples):
            normalizer = EmpiricalCDFNormalizer().fit(np.asarray(reference))
            probability = float(normalizer.transform([value])[0])
            if self.tail == "high":
                score = probability
            elif self.tail == "low":
                score = 1.0 - probability
            else:
                score = 2.0 * abs(probability - 0.5)
            scores[index] = score
            if score <= self.update_guard_score:
                reference.append(float(value))
                reference = reference[-self.reference_window :]
                updates += 1
            else:
                frozen += 1
        self.last_update_count_ = updates
        self.last_frozen_count_ = frozen
        return scores

    def predict(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        scores = self.score_samples(values)
        return AlarmOnOffDelay(
            1.0 - self.tail_probability, "high", self.delay
        ).predict(scores)


class AdaptedBookUnivariateSuite:
    """Run all four Chapter 2 mechanisms on an empirical-CDF alarm score."""

    def __init__(
        self,
        *,
        tail: Tail = "two_sided",
        threshold_candidates: int = 13,
        delays: Sequence[int] = (1, 2, 3, 4, 5),
        targets: tuple[float, float, float] = (0.05, 0.05, 10.0),
        weights: tuple[float, float, float] = (1.0, 1.0, 0.25),
        posterior_confidence: float = 0.95,
        app_probability_weight: float = 0.5,
    ) -> None:
        if threshold_candidates < 3:
            raise ValueError("threshold_candidates must be at least three")
        if not delays:
            raise ValueError("delays must not be empty")
        self.tail = tail
        self.threshold_candidates = int(threshold_candidates)
        self.delays = tuple(int(value) for value in delays)
        self.targets = tuple(float(value) for value in targets)
        self.weights = tuple(float(value) for value in weights)
        self.posterior_confidence = float(posterior_confidence)
        self.app_probability_weight = float(app_probability_weight)

    def _score(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        probabilities = self.normalizer_.transform(values)
        if self.tail == "high":
            return probabilities
        if self.tail == "low":
            return 1.0 - probabilities
        if self.tail == "two_sided":
            return 2.0 * np.abs(probabilities - 0.5)
        raise ValueError("tail must be high, low, or two_sided")

    def fit(
        self,
        normal: Sequence[float] | np.ndarray,
        abnormal_calibration: Sequence[float] | np.ndarray,
    ) -> "AdaptedBookUnivariateSuite":
        normal_values = _one_dimensional(normal, "normal")
        abnormal_values = _one_dimensional(
            abnormal_calibration, "abnormal_calibration"
        )
        self.normalizer_ = EmpiricalCDFNormalizer().fit(normal_values)
        normal_score = self._score(normal_values)
        abnormal_score = self._score(abnormal_values)
        thresholds = np.unique(
            np.quantile(
                np.r_[normal_score, abnormal_score],
                np.linspace(0.15, 0.85, self.threshold_candidates),
            )
        )
        if len(thresholds) < 2:
            support = np.unique(np.r_[normal_score, abnormal_score])
            if len(support) >= 2:
                indices = np.unique(
                    np.linspace(0, len(support) - 1, self.threshold_candidates).astype(int)
                )
                thresholds = support[indices]
        if len(thresholds) < 2:
            raise ValueError("adapted score has fewer than two observed support points")

        iid = design_iid_delay_timer(
            normal_score,
            abnormal_score,
            thresholds=thresholds,
            delays=self.delays,
            direction="high",
            targets=self.targets,
            weights=self.weights,
        )
        non_iid = design_non_iid_delay_timer(
            normal_score,
            abnormal_score,
            thresholds=thresholds,
            delays=self.delays,
            direction="high",
            target_far=self.targets[0],
            target_mar=self.targets[1],
            far_weight=self.weights[0] / max(self.weights[0] + self.weights[1], 1e-12),
            confidence=self.posterior_confidence,
        )
        base_threshold = float(np.quantile(normal_score, 0.95))
        episodes = alarm_episode_metrics(normal_score, base_threshold, "high")
        if len(episodes.durations) >= 2:
            deadband_shape = deadband_index(
                normal_score, base_threshold, direction="high"
            )
            maximum_width = max(
                float(np.max(episodes.deviations)), np.finfo(float).eps
            )
            deadband = design_deadband_width(
                episodes.deviations,
                maximum_width=maximum_width,
                target_remaining_probability=self.targets[0],
                confidence=self.posterior_confidence,
            )
            deadband_width = deadband.width
            deadband_suitable = deadband_shape.suitable
            deadband_status = "designed"
        else:
            deadband_width = 0.0
            deadband_suitable = False
            deadband_status = "denied_fewer_than_two_normal_alarm_episodes"
        activation_threshold = base_threshold + deadband_width
        combined = np.r_[normal_score, abnormal_score]
        minimum_state_samples = max(20, len(combined) // 20)
        app_thresholds = thresholds[
            (thresholds > float(np.min(combined)))
            & (thresholds < float(np.max(combined)))
        ]
        if not len(app_thresholds):
            support = np.unique(combined)
            app_thresholds = (support[:-1] + support[1:]) / 2.0
        app = select_alarm_probability_threshold(
            combined,
            app_thresholds,
            minimum_state_samples=minimum_state_samples,
            probability_weight=self.app_probability_weight,
        )
        self.models_ = {
            "book_2_1_iid_delay_timer": AlarmOnOffDelay(
                iid.threshold, "high", iid.delay
            ),
            "book_2_2_non_iid_delay_timer": AlarmOnOffDelay(
                non_iid.threshold, "high", non_iid.delay
            ),
            "book_2_3_non_iid_deadband": ThresholdDelayDeadband(
                activation_threshold, "high", delay=1, deadband=deadband_width
            ),
            "book_2_4_alarm_probability_plot": ThresholdDelayDeadband(
                app.threshold, "high"
            ),
        }
        self.design_summary_ = {
            "book_2_1_iid_delay_timer": {
                "threshold": iid.threshold,
                "delay": iid.delay,
                "design_loss": iid.loss,
            },
            "book_2_2_non_iid_delay_timer": {
                "threshold": non_iid.threshold,
                "delay": non_iid.delay,
                "design_loss": non_iid.loss,
                "zero_event_fallback": non_iid.zero_event_fallback,
            },
            "book_2_3_non_iid_deadband": {
                "base_threshold": base_threshold,
                "activation_threshold": activation_threshold,
                "width": deadband_width,
                "suitable": deadband_suitable,
                "status": deadband_status,
            },
            "book_2_4_alarm_probability_plot": {
                "threshold": app.threshold,
                "score": app.score,
                "states": len(app.plot.transition_matrix),
            },
        }
        return self

    def predict(
        self, values: Sequence[float] | np.ndarray
    ) -> dict[str, np.ndarray]:
        if not hasattr(self, "models_"):
            raise RuntimeError("AdaptedBookUnivariateSuite is not fitted")
        score = self._score(values)
        return {name: model.predict(score) for name, model in self.models_.items()}


@dataclass(frozen=True)
class AdaptiveRouterDecision:
    status: Literal["static", "adapt", "reject_univariate"]
    feature_index: int
    feature_name: str
    direction: Literal["high", "low"]
    selected_model: str | None
    calibration_applicability: CalibrationApplicability
    reason: str


class AdaptiveUnivariateAlarmRouter:
    """Select a leakage-safe adapter or explicitly deny univariate scoring."""

    def __init__(
        self,
        *,
        applicability_thresholds: ApplicabilityThresholds | None = None,
        reference_windows: Sequence[int] = (128, 256, 512, 1024),
        delays: Sequence[int] = (1, 2, 3, 4, 5),
        block_size: int = 60,
        tail_probability: float = 0.05,
    ) -> None:
        self.applicability_thresholds = (
            applicability_thresholds or ApplicabilityThresholds()
        )
        self.reference_windows = tuple(int(value) for value in reference_windows)
        self.delays = tuple(int(value) for value in delays)
        self.block_size = int(block_size)
        self.tail_probability = float(tail_probability)

    def fit(
        self,
        normal: Sequence[Sequence[float]] | np.ndarray,
        abnormal_calibration: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> "AdaptiveUnivariateAlarmRouter":
        normal_values = _matrix(normal, "normal")
        abnormal_values = _matrix(abnormal_calibration, "abnormal_calibration")
        selector = FeatureStabilitySelector(
            chronological_blocks=self.applicability_thresholds.chronological_blocks
        ).fit(normal_values, abnormal_values, feature_names=feature_names)
        normal_feature = selector.transform(normal_values)
        abnormal_feature = selector.transform(abnormal_values)
        applicability = assess_univariate_calibration(
            normal_feature,
            abnormal_feature,
            thresholds=self.applicability_thresholds,
        )
        selected_model: str | None = None
        reason: str
        self.model_ = None
        if applicability.status == "reject_univariate":
            reason = "calibration gate requires multivariate fallback"
        elif applicability.status == "adapt":
            self.model_ = BlockCalibratedECDFAlarm(
                tail_probability=self.tail_probability,
                tail=selector.direction_,
                reference_windows=self.reference_windows,
                delays=self.delays,
                block_size=self.block_size,
            ).fit(normal_feature)
            selected_model = "block_calibrated_ecdf"
            reason = "distribution or temporal adaptation required"
        else:
            self.model_ = EmpiricalCDFAlarm(
                tail_probability=self.tail_probability,
                tail=selector.direction_,
                delay=1,
            ).fit(normal_feature)
            selected_model = "static_ecdf"
            reason = "calibration distribution supports static transfer"
        self.selector_ = selector
        self.decision_ = AdaptiveRouterDecision(
            status=applicability.status,
            feature_index=selector.feature_index_,
            feature_name=selector.feature_name_,
            direction=selector.direction_,
            selected_model=selected_model,
            calibration_applicability=applicability,
            reason=reason,
        )
        return self

    def predict(
        self, values: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "decision_"):
            raise RuntimeError("AdaptiveUnivariateAlarmRouter is not fitted")
        if self.model_ is None:
            raise RuntimeError("univariate prediction denied by calibration gate")
        return self.model_.predict(self.selector_.transform(values))


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
