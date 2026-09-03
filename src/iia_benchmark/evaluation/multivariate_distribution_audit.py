"""Leakage-safe distribution diagnostics for multivariate alarm transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.stats import ks_2samp, rankdata, wasserstein_distance


def _matrix(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 2 or result.shape[0] < 4 or result.shape[1] < 2:
        raise ValueError(f"{name} must contain at least four rows and two features")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _lag_one(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=float)
    for index in range(values.shape[1]):
        left = values[:-1, index]
        right = values[1:, index]
        if np.std(left) > 1e-12 and np.std(right) > 1e-12:
            result[index] = np.corrcoef(left, right)[0, 1]
    return result


def _condition(covariance: np.ndarray) -> float:
    value = float(np.linalg.cond(covariance))
    return value if np.isfinite(value) else float(np.finfo(float).max)


def _effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 1e-15] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _auc(reference: np.ndarray, candidate: np.ndarray) -> float:
    combined = np.r_[reference, candidate]
    ranks = rankdata(combined, method="average")
    n0 = len(reference)
    n1 = len(candidate)
    statistic = float(np.sum(ranks[n0:]) - n1 * (n1 + 1) / 2.0)
    return statistic / (n0 * n1)


@dataclass(frozen=True)
class MultivariateDistributionShift:
    reference_samples: int
    candidate_samples: int
    features: int
    per_feature_ks_median: float
    per_feature_ks_maximum: float
    wasserstein_over_reference_iqr_median: float
    wasserstein_over_reference_iqr_maximum: float
    standardized_median_shift_l2: float
    standardized_median_shift_maximum: float
    covariance_relative_frobenius_shift: float
    maximum_absolute_correlation_shift: float
    reference_covariance_condition: float
    candidate_covariance_condition: float
    reference_effective_rank: float
    candidate_effective_rank: float
    candidate_absolute_lag_one_median: float
    candidate_absolute_lag_one_maximum: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def multivariate_distribution_shift(
    reference: Iterable[Iterable[float]],
    candidate: Iterable[Iterable[float]],
) -> MultivariateDistributionShift:
    """Describe marginal, covariance, rank, and temporal transfer shift."""

    first = _matrix(reference, "reference")
    second = _matrix(candidate, "candidate")
    if first.shape[1] != second.shape[1]:
        raise ValueError("reference and candidate feature counts differ")
    median = np.median(first, axis=0)
    iqr = np.quantile(first, 0.75, axis=0) - np.quantile(first, 0.25, axis=0)
    standard = np.std(first, axis=0, ddof=1)
    scale = np.where(iqr > 1e-12, iqr / 1.349, standard)
    scale[scale <= 1e-12] = 1.0
    normalized_first = (first - median) / scale
    normalized_second = (second - median) / scale
    ks = np.asarray(
        [ks_2samp(first[:, i], second[:, i]).statistic for i in range(first.shape[1])]
    )
    wasserstein = np.asarray(
        [
            wasserstein_distance(first[:, i], second[:, i])
            / max(iqr[i], 1e-12)
            for i in range(first.shape[1])
        ]
    )
    median_shift = np.abs(np.median(normalized_second, axis=0))
    covariance_first = np.atleast_2d(np.cov(normalized_first, rowvar=False))
    covariance_second = np.atleast_2d(np.cov(normalized_second, rowvar=False))
    denominator = max(float(np.linalg.norm(covariance_first, ord="fro")), 1e-12)
    covariance_shift = float(
        np.linalg.norm(covariance_second - covariance_first, ord="fro") / denominator
    )
    correlation_first = np.nan_to_num(np.corrcoef(normalized_first, rowvar=False))
    correlation_second = np.nan_to_num(np.corrcoef(normalized_second, rowvar=False))
    correlation_shift = float(np.max(np.abs(correlation_second - correlation_first)))
    lag = np.abs(_lag_one(second))
    return MultivariateDistributionShift(
        reference_samples=len(first),
        candidate_samples=len(second),
        features=first.shape[1],
        per_feature_ks_median=float(np.median(ks)),
        per_feature_ks_maximum=float(np.max(ks)),
        wasserstein_over_reference_iqr_median=float(np.median(wasserstein)),
        wasserstein_over_reference_iqr_maximum=float(np.max(wasserstein)),
        standardized_median_shift_l2=float(np.linalg.norm(median_shift)),
        standardized_median_shift_maximum=float(np.max(median_shift)),
        covariance_relative_frobenius_shift=covariance_shift,
        maximum_absolute_correlation_shift=correlation_shift,
        reference_covariance_condition=_condition(covariance_first),
        candidate_covariance_condition=_condition(covariance_second),
        reference_effective_rank=_effective_rank(covariance_first),
        candidate_effective_rank=_effective_rank(covariance_second),
        candidate_absolute_lag_one_median=float(np.median(lag)),
        candidate_absolute_lag_one_maximum=float(np.max(lag)),
    )


@dataclass(frozen=True)
class MultivariateApplicabilityThresholds:
    normal_ks_adaptation: float = 0.20
    normal_median_shift_adaptation: float = 0.50
    normal_covariance_shift_adaptation: float = 0.50
    autocorrelation_block_calibration: float = 0.80
    minimum_block_auc: float = 0.60
    chronological_blocks: int = 3


@dataclass(frozen=True)
class MultivariateApplicability:
    status: Literal["static", "adapt", "reject_multivariate"]
    internal_normal_shift: MultivariateDistributionShift
    calibration_auc: float
    abnormal_block_auc: tuple[float, ...]
    recommended_adapters: tuple[str, ...]
    reasons: tuple[str, ...]
    scope: str = "training_and_calibration_only"

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["internal_normal_shift"] = self.internal_normal_shift.as_dict()
        return result


def assess_multivariate_calibration(
    normal: Iterable[Iterable[float]],
    abnormal_calibration: Iterable[Iterable[float]],
    *,
    thresholds: MultivariateApplicabilityThresholds | None = None,
) -> MultivariateApplicability:
    """Route without reading held-out normal or abnormal evaluation partitions."""

    limits = thresholds or MultivariateApplicabilityThresholds()
    normal_values = _matrix(normal, "normal")
    abnormal_values = _matrix(abnormal_calibration, "abnormal_calibration")
    if normal_values.shape[1] != abnormal_values.shape[1]:
        raise ValueError("normal and abnormal feature counts differ")
    if limits.chronological_blocks < 2:
        raise ValueError("chronological_blocks must be at least two")
    split = len(normal_values) // 2
    internal = multivariate_distribution_shift(
        normal_values[:split], normal_values[split:]
    )
    location = np.median(normal_values, axis=0)
    scale = np.std(normal_values, axis=0, ddof=1)
    scale[scale <= 1e-12] = 1.0
    direction = (np.median(abnormal_values, axis=0) - location) / scale
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.ones(normal_values.shape[1]) / np.sqrt(normal_values.shape[1])
    else:
        direction /= norm
    normal_score = ((normal_values - location) / scale) @ direction
    blocks = np.array_split(abnormal_values, limits.chronological_blocks)
    block_auc = tuple(
        float(_auc(normal_score, ((block - location) / scale) @ direction))
        for block in blocks
    )
    calibration_auc = float(
        _auc(normal_score, ((abnormal_values - location) / scale) @ direction)
    )
    reasons: list[str] = []
    adapters: list[str] = ["robust_scaling", "covariance_shrinkage"]
    if min(block_auc) < limits.minimum_block_auc:
        reasons.append("weak_or_unstable_calibration_separation")
        return MultivariateApplicability(
            status="reject_multivariate",
            internal_normal_shift=internal,
            calibration_auc=calibration_auc,
            abnormal_block_auc=block_auc,
            recommended_adapters=("dynamic_residual_or_task_specific_model",),
            reasons=tuple(reasons),
        )
    adapt = False
    if internal.per_feature_ks_median > limits.normal_ks_adaptation:
        adapt = True
        reasons.append("internal_normal_marginal_shift")
    if (
        internal.standardized_median_shift_maximum
        > limits.normal_median_shift_adaptation
    ):
        adapt = True
        reasons.append("internal_normal_location_shift")
    if (
        internal.covariance_relative_frobenius_shift
        > limits.normal_covariance_shift_adaptation
    ):
        adapt = True
        reasons.append("internal_normal_covariance_shift")
    if (
        internal.candidate_absolute_lag_one_median
        > limits.autocorrelation_block_calibration
    ):
        adapt = True
        reasons.append("strong_temporal_dependence")
    if adapt:
        adapters.extend(["recent_reference", "block_calibration", "delay"])
    return MultivariateApplicability(
        status="adapt" if adapt else "static",
        internal_normal_shift=internal,
        calibration_auc=calibration_auc,
        abnormal_block_auc=block_auc,
        recommended_adapters=tuple(adapters),
        reasons=tuple(reasons or ["static_transfer_supported"]),
    )
