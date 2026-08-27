from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


Direction = Literal["high", "low"]


@dataclass(frozen=True)
class ThresholdDelayDeadband:
    """Stateful alarm generator with an on-delay and a hysteresis deadband.

    ``delay=1`` is a basic threshold alarm.  For a high alarm, activation needs
    ``delay`` consecutive values at or above ``threshold`` and clearance needs a
    value strictly below ``threshold - deadband``.  The low-alarm definition is
    symmetric.
    """

    threshold: float
    direction: Direction = "high"
    delay: int = 1
    deadband: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in ("high", "low"):
            raise ValueError("direction must be 'high' or 'low'")
        if self.delay < 1:
            raise ValueError("delay must be at least 1")
        if self.deadband < 0:
            raise ValueError("deadband must be non-negative")

    def predict(self, values: Iterable[float]) -> np.ndarray:
        samples = np.asarray(list(values), dtype=float)
        result = np.zeros(len(samples), dtype=np.int8)
        active = False
        consecutive = 0
        for index, value in enumerate(samples):
            beyond = value >= self.threshold if self.direction == "high" else value <= self.threshold
            cleared = (
                value < self.threshold - self.deadband
                if self.direction == "high"
                else value > self.threshold + self.deadband
            )
            if active:
                if cleared:
                    active = False
                    consecutive = 0
            elif beyond:
                consecutive += 1
                if consecutive >= self.delay:
                    active = True
            else:
                consecutive = 0
            result[index] = int(active)
        return result


@dataclass(frozen=True)
class AlarmDesignResult:
    model: ThresholdDelayDeadband
    false_alarm_rate: float
    missed_alarm_rate: float
    average_alarm_delay: float
    loss: float


def _episode_delays(abnormal: np.ndarray, alarm: np.ndarray) -> list[int]:
    starts = np.flatnonzero(abnormal & ~np.r_[False, abnormal[:-1]])
    delays: list[int] = []
    for start in starts:
        end_candidates = np.flatnonzero(~abnormal[start:])
        end = start + int(end_candidates[0]) if len(end_candidates) else len(abnormal)
        hits = np.flatnonzero(alarm[start:end])
        delays.append(int(hits[0]) if len(hits) else end - start)
    return delays


def evaluate_alarm_design(
    abnormal: Iterable[bool], alarm: Iterable[int]
) -> tuple[float, float, float]:
    truth = np.asarray(list(abnormal), dtype=bool)
    prediction = np.asarray(list(alarm), dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("abnormal and alarm must be non-empty one-dimensional arrays")
    normal_count = int((~truth).sum())
    abnormal_count = int(truth.sum())
    far = float((prediction & ~truth).sum() / normal_count) if normal_count else 0.0
    mar = float((~prediction & truth).sum() / abnormal_count) if abnormal_count else 0.0
    delays = _episode_delays(truth, prediction)
    aad = float(np.mean(delays)) if delays else 0.0
    return far, mar, aad


def design_alarm(
    values: Iterable[float],
    abnormal: Iterable[bool],
    *,
    thresholds: Iterable[float],
    delays: Iterable[int] = (1,),
    deadbands: Iterable[float] = (0.0,),
    direction: Direction = "high",
    targets: tuple[float, float, float] = (0.05, 0.05, 10.0),
    weights: tuple[float, float, float] = (1.0, 1.0, 0.25),
) -> AlarmDesignResult:
    """Grid-search alarm parameters using the book's FAR/MAR/AAD trade-off."""
    samples = np.asarray(list(values), dtype=float)
    truth = np.asarray(list(abnormal), dtype=bool)
    if len(samples) != len(truth) or not len(samples):
        raise ValueError("values and abnormal must be non-empty and equal length")
    target = np.maximum(np.asarray(targets, dtype=float), 1e-12)
    weight = np.asarray(weights, dtype=float)
    candidates: list[AlarmDesignResult] = []
    for threshold in thresholds:
        for delay in delays:
            for deadband in deadbands:
                model = ThresholdDelayDeadband(
                    float(threshold), direction, int(delay), float(deadband)
                )
                alarm = model.predict(samples)
                far, mar, aad = evaluate_alarm_design(truth, alarm)
                observed = np.asarray((far, mar, aad), dtype=float)
                loss = float(np.sum(weight * observed / target))
                candidates.append(AlarmDesignResult(model, far, mar, aad, loss))
    if not candidates:
        raise ValueError("parameter grids must contain at least one candidate")
    return min(
        candidates,
        key=lambda result: (
            result.loss,
            result.missed_alarm_rate,
            result.false_alarm_rate,
            result.average_alarm_delay,
        ),
    )

