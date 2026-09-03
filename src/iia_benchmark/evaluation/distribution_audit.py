"""Distribution and applicability audits for univariate alarm transfer.

The calibration gate in this module is deliberately separate from the held-out
transfer audit.  The former may be used to route a model before evaluation; the
latter is post-hoc benchmark evidence and must never be used for tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.stats import kurtosis, ks_2samp, rankdata, skew, wasserstein_distance


AlarmDirection = Literal["high", "low"]


def _vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _scale(values: np.ndarray) -> float:
    scale = float(np.std(values, ddof=1))
    if scale <= 1e-12:
        raise ValueError("normal reference is constant")
    return scale


def _alarm_sign(direction: AlarmDirection) -> float:
    if direction == "high":
        return 1.0
    if direction == "low":
        return -1.0
    raise ValueError("direction must be 'high' or 'low'")


def _direction(normal: np.ndarray, abnormal: np.ndarray) -> AlarmDirection:
    return "high" if np.median(abnormal) >= np.median(normal) else "low"


def _auc(normal_score: np.ndarray, abnormal_score: np.ndarray) -> float:
    combined = np.r_[normal_score, abnormal_score]
    ranks = rankdata(combined, method="average")
    normal_count = len(normal_score)
    abnormal_count = len(abnormal_score)
    abnormal_rank_sum = float(np.sum(ranks[normal_count:]))
    statistic = abnormal_rank_sum - abnormal_count * (abnormal_count + 1) / 2.0
    return float(statistic / (normal_count * abnormal_count))


def _lag_one(values: np.ndarray) -> float:
    left = values[:-1]
    right = values[1:]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _true_run_summary(values: np.ndarray) -> tuple[int, float, int]:
    states = np.asarray(values, dtype=bool)
    starts = np.flatnonzero(states & np.r_[True, ~states[:-1]])
    stops = np.flatnonzero(states & np.r_[~states[1:], True])
    lengths = stops - starts + 1
    if not len(lengths):
        return 0, 0.0, 0
    return int(len(lengths)), float(np.mean(lengths)), int(np.max(lengths))


@dataclass(frozen=True)
class DistributionShift:
    """Effect-size description of a candidate distribution versus a reference."""

    ks: float
    wasserstein: float
    wasserstein_over_reference_iqr: float
    standardized_median_shift: float


@dataclass(frozen=True)
class TemporalProfile:
    samples: int
    lag_one_autocorrelation: float
    skewness: float
    excess_kurtosis: float


@dataclass(frozen=True)
class ThresholdTransfer:
    threshold: float
    normal_train_exceedance_rate: float
    normal_evaluation_exceedance_rate: float
    normal_evaluation_alarm_runs: int
    normal_evaluation_mean_run_length: float
    normal_evaluation_maximum_run_length: int


@dataclass(frozen=True)
class UnivariateDistributionAudit:
    """Held-out transfer evidence; this object is not a deployment-time gate."""

    direction: AlarmDirection
    samples: dict[str, int]
    normal_train_to_evaluation: DistributionShift
    abnormal_calibration_to_evaluation: DistributionShift
    calibration_separation_sd: float
    evaluation_separation_sd: float
    calibration_auc: float
    evaluation_auc: float
    alarm_direction_consistent: bool
    normal_train_temporal: TemporalProfile
    normal_evaluation_temporal: TemporalProfile
    evaluation_abnormal_prevalence: float
    threshold_transfer: ThresholdTransfer | None
    scope: str = "held_out_posthoc_diagnostic_not_for_routing"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ApplicabilityThresholds:
    normal_ks_adaptation: float = 0.20
    normal_median_shift_sd_adaptation: float = 0.50
    autocorrelation_block_calibration: float = 0.80
    minimum_block_auc: float = 0.60
    minimum_direction_consistency: float = 2.0 / 3.0
    chronological_blocks: int = 3


@dataclass(frozen=True)
class CalibrationApplicability:
    """Leakage-safe decision based only on training and calibration partitions."""

    status: Literal["static", "adapt", "reject_univariate"]
    direction: AlarmDirection
    normal_internal_shift: DistributionShift
    normal_lag_one_autocorrelation: float
    calibration_auc: float
    abnormal_block_auc: tuple[float, ...]
    abnormal_direction_consistency: float
    recommended_adapters: tuple[str, ...]
    reasons: tuple[str, ...]
    scope: str = "training_and_calibration_only"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def distribution_shift(
    reference: np.ndarray | list[float],
    candidate: np.ndarray | list[float],
    *,
    direction: AlarmDirection = "high",
) -> DistributionShift:
    """Return distance and signed alarm-direction location shift."""

    reference_array = _vector(reference, "reference")
    candidate_array = _vector(candidate, "candidate")
    scale = _scale(reference_array)
    reference_iqr = float(
        np.quantile(reference_array, 0.75) - np.quantile(reference_array, 0.25)
    )
    denominator = reference_iqr if reference_iqr > 1e-12 else scale
    sign = _alarm_sign(direction)
    return DistributionShift(
        ks=float(ks_2samp(reference_array, candidate_array).statistic),
        wasserstein=float(wasserstein_distance(reference_array, candidate_array)),
        wasserstein_over_reference_iqr=float(
            wasserstein_distance(reference_array, candidate_array) / denominator
        ),
        standardized_median_shift=float(
            sign
            * (np.median(candidate_array) - np.median(reference_array))
            / scale
        ),
    )


def temporal_profile(values: np.ndarray | list[float]) -> TemporalProfile:
    array = _vector(values, "values")
    return TemporalProfile(
        samples=int(len(array)),
        lag_one_autocorrelation=_lag_one(array),
        skewness=float(skew(array, bias=False)),
        excess_kurtosis=float(kurtosis(array, bias=False)),
    )


def audit_univariate_partitions(
    normal_train: np.ndarray | list[float],
    normal_evaluation: np.ndarray | list[float],
    abnormal_calibration: np.ndarray | list[float],
    abnormal_evaluation: np.ndarray | list[float],
    *,
    direction: AlarmDirection | None = None,
    threshold: float | None = None,
) -> UnivariateDistributionAudit:
    """Characterize frozen partitions after a feature has been selected.

    Evaluation arrays are intentionally accepted here because this function is
    for benchmark diagnosis.  Use :func:`assess_univariate_calibration` for a
    leakage-safe deployment decision.
    """

    normal_fit = _vector(normal_train, "normal_train")
    normal_test = _vector(normal_evaluation, "normal_evaluation")
    abnormal_fit = _vector(abnormal_calibration, "abnormal_calibration")
    abnormal_test = _vector(abnormal_evaluation, "abnormal_evaluation")
    selected_direction = direction or _direction(normal_fit, abnormal_fit)
    sign = _alarm_sign(selected_direction)
    scale = _scale(normal_fit)

    calibration_delta = float(np.median(abnormal_fit) - np.median(normal_fit))
    evaluation_delta = float(np.median(abnormal_test) - np.median(normal_test))
    threshold_report = None
    if threshold is not None:
        oriented_threshold = sign * float(threshold)
        train_exceedance = sign * normal_fit >= oriented_threshold
        evaluation_exceedance = sign * normal_test >= oriented_threshold
        count, mean_length, maximum_length = _true_run_summary(evaluation_exceedance)
        threshold_report = ThresholdTransfer(
            threshold=float(threshold),
            normal_train_exceedance_rate=float(np.mean(train_exceedance)),
            normal_evaluation_exceedance_rate=float(np.mean(evaluation_exceedance)),
            normal_evaluation_alarm_runs=count,
            normal_evaluation_mean_run_length=mean_length,
            normal_evaluation_maximum_run_length=maximum_length,
        )

    return UnivariateDistributionAudit(
        direction=selected_direction,
        samples={
            "normal_train": int(len(normal_fit)),
            "normal_evaluation": int(len(normal_test)),
            "abnormal_calibration": int(len(abnormal_fit)),
            "abnormal_evaluation": int(len(abnormal_test)),
        },
        normal_train_to_evaluation=distribution_shift(
            normal_fit, normal_test, direction=selected_direction
        ),
        abnormal_calibration_to_evaluation=distribution_shift(
            abnormal_fit, abnormal_test, direction=selected_direction
        ),
        calibration_separation_sd=float(sign * calibration_delta / scale),
        evaluation_separation_sd=float(sign * evaluation_delta / scale),
        calibration_auc=_auc(sign * normal_fit, sign * abnormal_fit),
        evaluation_auc=_auc(sign * normal_test, sign * abnormal_test),
        alarm_direction_consistent=bool(
            np.sign(calibration_delta) == np.sign(evaluation_delta)
            and abs(calibration_delta) > 1e-12
            and abs(evaluation_delta) > 1e-12
        ),
        normal_train_temporal=temporal_profile(normal_fit),
        normal_evaluation_temporal=temporal_profile(normal_test),
        evaluation_abnormal_prevalence=float(
            len(abnormal_test) / (len(normal_test) + len(abnormal_test))
        ),
        threshold_transfer=threshold_report,
    )


def assess_univariate_calibration(
    normal_train: np.ndarray | list[float],
    abnormal_calibration: np.ndarray | list[float],
    *,
    thresholds: ApplicabilityThresholds | None = None,
) -> CalibrationApplicability:
    """Route a selected feature without consulting its final evaluation set."""

    normal = _vector(normal_train, "normal_train")
    abnormal = _vector(abnormal_calibration, "abnormal_calibration")
    policy = thresholds or ApplicabilityThresholds()
    if policy.chronological_blocks < 2:
        raise ValueError("chronological_blocks must be at least two")

    split = len(normal) // 2
    if split < 2 or len(normal) - split < 2:
        raise ValueError("normal_train is too short for an internal chronological split")
    selected_direction = _direction(normal, abnormal)
    sign = _alarm_sign(selected_direction)
    normal_shift = distribution_shift(
        normal[:split], normal[split:], direction=selected_direction
    )
    blocks = [block for block in np.array_split(abnormal, policy.chronological_blocks) if len(block) >= 2]
    if len(blocks) != policy.chronological_blocks:
        raise ValueError("abnormal_calibration is too short for chronological blocks")

    normal_score = sign * normal
    auc_values = tuple(_auc(normal_score, sign * block) for block in blocks)
    global_delta = sign * float(np.median(abnormal) - np.median(normal))
    block_deltas = [sign * float(np.median(block) - np.median(normal)) for block in blocks]
    direction_consistency = float(np.mean([delta > 1e-12 for delta in block_deltas]))
    adapters: list[str] = []
    reasons: list[str] = []

    if (
        normal_shift.ks >= policy.normal_ks_adaptation
        or abs(normal_shift.standardized_median_shift)
        >= policy.normal_median_shift_sd_adaptation
    ):
        adapters.extend(["robust_ecdf", "regime_conditional_threshold"])
        reasons.append("internal_normal_distribution_shift")
    lag_one = _lag_one(normal)
    if abs(lag_one) >= policy.autocorrelation_block_calibration:
        adapters.extend(["block_calibration", "event_level_metrics"])
        reasons.append("strong_temporal_dependence")
    if direction_consistency < policy.minimum_direction_consistency or global_delta <= 1e-12:
        adapters.append("two_sided_change_detection")
        reasons.append("unstable_abnormal_direction")
    if min(auc_values) < policy.minimum_block_auc:
        adapters.append("multivariate_fallback")
        reasons.append("insufficient_worst_block_separability")

    unique_adapters = tuple(dict.fromkeys(adapters))
    reject = bool(
        "unstable_abnormal_direction" in reasons
        or "insufficient_worst_block_separability" in reasons
    )
    status: Literal["static", "adapt", "reject_univariate"]
    if reject:
        status = "reject_univariate"
    elif unique_adapters:
        status = "adapt"
    else:
        status = "static"
    return CalibrationApplicability(
        status=status,
        direction=selected_direction,
        normal_internal_shift=normal_shift,
        normal_lag_one_autocorrelation=lag_one,
        calibration_auc=_auc(normal_score, sign * abnormal),
        abnormal_block_auc=auc_values,
        abnormal_direction_consistency=direction_consistency,
        recommended_adapters=unique_adapters,
        reasons=tuple(reasons),
    )
