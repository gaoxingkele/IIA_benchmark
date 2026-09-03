"""Event-level alarm metrics and dependent-data uncertainty estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

from .metrics import binary_alarm_metrics


def _binary(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return result


def _run_lengths(states: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(states & np.r_[True, ~states[:-1]])
    stops = np.flatnonzero(states & np.r_[~states[1:], True])
    return (stops - starts + 1).astype(int)


@dataclass(frozen=True)
class AlarmEventMetrics:
    false_alarm_events: int
    false_alarm_events_per_hour: float
    normal_alarm_mean_duration_samples: float
    normal_alarm_maximum_duration_samples: int
    abnormal_event_recall: float
    detection_delay_samples: int
    detection_delay_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float
    bootstrap_mean: float
    standard_error: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BlockBootstrapAlarmReport:
    block_size: int
    draws: int
    confidence: float
    seed: int
    metrics: dict[str, BootstrapInterval]
    resampling_policy: str = "moving blocks sampled separately within normal and abnormal partitions"

    def as_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "draws": self.draws,
            "confidence": self.confidence,
            "seed": self.seed,
            "metrics": {name: value.as_dict() for name, value in self.metrics.items()},
            "resampling_policy": self.resampling_policy,
        }


@dataclass(frozen=True)
class BlockEventRatePosterior:
    """Beta posterior for the probability that a time block contains an alarm."""

    events: int
    blocks: int
    block_size: int
    prior_alpha: float
    prior_beta: float
    posterior_mean: float
    lower: float
    upper: float
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def alarm_event_metrics(
    normal_alarm: Sequence[int] | np.ndarray,
    abnormal_alarm: Sequence[int] | np.ndarray,
    *,
    sample_period_seconds: float,
) -> AlarmEventMetrics:
    """Score a normal segment followed by one abnormal episode."""

    normal = _binary(normal_alarm, "normal_alarm")
    abnormal = _binary(abnormal_alarm, "abnormal_alarm")
    if sample_period_seconds <= 0:
        raise ValueError("sample_period_seconds must be positive")
    false_runs = _run_lengths(normal)
    normal_hours = len(normal) * sample_period_seconds / 3600.0
    hits = np.flatnonzero(abnormal)
    delay = int(hits[0]) if len(hits) else len(abnormal)
    return AlarmEventMetrics(
        false_alarm_events=int(len(false_runs)),
        false_alarm_events_per_hour=float(len(false_runs) / normal_hours),
        normal_alarm_mean_duration_samples=(
            float(np.mean(false_runs)) if len(false_runs) else 0.0
        ),
        normal_alarm_maximum_duration_samples=(
            int(np.max(false_runs)) if len(false_runs) else 0
        ),
        abnormal_event_recall=float(bool(len(hits))),
        detection_delay_samples=delay,
        detection_delay_seconds=float(delay * sample_period_seconds),
    )


def _moving_block_resample(
    values: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    effective = min(block_size, len(values))
    maximum_start = len(values) - effective
    blocks = int(np.ceil(len(values) / effective))
    starts = rng.integers(0, maximum_start + 1, blocks)
    result = np.concatenate([values[start : start + effective] for start in starts])
    return result[: len(values)]


def block_bootstrap_alarm_metrics(
    normal_alarm: Sequence[int] | np.ndarray,
    abnormal_alarm: Sequence[int] | np.ndarray,
    *,
    block_size: int = 60,
    draws: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
) -> BlockBootstrapAlarmReport:
    """Estimate uncertainty without treating adjacent samples as IID."""

    normal = _binary(normal_alarm, "normal_alarm")
    abnormal = _binary(abnormal_alarm, "abnormal_alarm")
    if block_size < 2:
        raise ValueError("block_size must be at least two")
    if draws < 20:
        raise ValueError("draws must be at least 20")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    truth = np.r_[np.zeros(len(normal), dtype=bool), np.ones(len(abnormal), dtype=bool)]
    point = binary_alarm_metrics(truth, np.r_[normal, abnormal])
    names = ("false_alarm_rate", "missed_alarm_rate", "precision", "recall", "f1")
    samples = {name: np.empty(draws, dtype=float) for name in names}
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        resampled_normal = _moving_block_resample(normal, block_size, rng)
        resampled_abnormal = _moving_block_resample(abnormal, block_size, rng)
        metrics = binary_alarm_metrics(
            truth, np.r_[resampled_normal, resampled_abnormal]
        )
        for name in names:
            samples[name][draw] = metrics[name]
    tail = (1.0 - confidence) / 2.0
    intervals = {
        name: BootstrapInterval(
            point_estimate=float(point[name]),
            bootstrap_mean=float(np.mean(values)),
            standard_error=float(np.std(values, ddof=1)),
            lower=float(np.quantile(values, tail)),
            upper=float(np.quantile(values, 1.0 - tail)),
        )
        for name, values in samples.items()
    }
    return BlockBootstrapAlarmReport(
        block_size=int(block_size),
        draws=int(draws),
        confidence=float(confidence),
        seed=int(seed),
        metrics=intervals,
    )


def block_event_rate_posterior(
    alarm: Sequence[int] | np.ndarray,
    *,
    block_size: int = 60,
    confidence: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BlockEventRatePosterior:
    """Estimate a block alarm-event probability, including the zero-event case."""

    values = _binary(alarm, "alarm")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive")
    blocks = tuple(
        values[start : start + block_size]
        for start in range(0, len(values), block_size)
    )
    events = int(sum(bool(np.any(block)) for block in blocks))
    posterior_alpha = prior_alpha + events
    posterior_beta = prior_beta + len(blocks) - events
    tail = (1.0 - confidence) / 2.0
    return BlockEventRatePosterior(
        events=events,
        blocks=len(blocks),
        block_size=int(block_size),
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        posterior_mean=float(posterior_alpha / (posterior_alpha + posterior_beta)),
        lower=float(beta_distribution.ppf(tail, posterior_alpha, posterior_beta)),
        upper=float(beta_distribution.ppf(1.0 - tail, posterior_alpha, posterior_beta)),
        confidence=float(confidence),
    )
