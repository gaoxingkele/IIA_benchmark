from __future__ import annotations

import numpy as np

from .schema import AlarmEpisode, AlarmEvent, ProcessRun


def make_synthetic_alarm_run(
    *, seed: int = 7, length: int = 1200, change_at: int = 700
) -> ProcessRun:
    """Generate a deterministic univariate run with an abrupt abnormal episode."""
    if not 20 < change_at < length:
        raise ValueError("change_at must leave normal and abnormal samples")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.55, length)
    values = np.empty(length, dtype=float)
    values[0] = noise[0]
    for index in range(1, length):
        target = 0.0 if index < change_at else 2.8
        values[index] = 0.72 * values[index - 1] + 0.28 * target + noise[index]
    abnormal = np.arange(length) >= change_at
    return ProcessRun(
        run_id="synthetic_step_fault",
        timestamps=np.arange(length, dtype=float),
        values=values,
        abnormal=abnormal,
        feature_names=("process_value",),
        root_cause="mean_shift",
    )


def make_synthetic_floods(*, seed: int = 11) -> tuple[AlarmEpisode, ...]:
    """Create small recurrent alarm floods with order jitter and nuisance tags."""
    rng = np.random.default_rng(seed)
    templates = {
        "feed_loss": ("F_LOW", "P_LOW", "LEVEL_LOW", "VALVE_OPEN"),
        "cooling_loss": ("T_HIGH", "P_HIGH", "CW_FLOW_LOW", "LEVEL_HIGH"),
    }
    episodes: list[AlarmEpisode] = []
    for label, tags in templates.items():
        for run in range(5):
            ordered = list(tags)
            if run % 2:
                ordered[1], ordered[2] = ordered[2], ordered[1]
            if run % 3 == 0:
                ordered.insert(2, "NUISANCE")
            events = tuple(
                AlarmEvent(timestamp=float(i * 20 + rng.integers(-3, 4)), tag=tag)
                for i, tag in enumerate(ordered)
            )
            episodes.append(
                AlarmEpisode(
                    episode_id=f"{label}_{run}",
                    events=events,
                    label=label,
                    root_cause=label,
                )
            )
    return tuple(episodes)


def make_synthetic_multivariate_run(
    *, seed: int = 19, length: int = 1200, change_at: int = 800
) -> ProcessRun:
    """Generate a correlated process whose normal operating relation later breaks."""
    if not 20 < change_at < length:
        raise ValueError("change_at must leave normal and abnormal samples")
    rng = np.random.default_rng(seed)
    feed = rng.normal(0.0, 1.0, length)
    pressure = 0.75 * feed + rng.normal(0.0, 0.35, length)
    temperature = -0.45 * feed + 0.55 * pressure + rng.normal(0.0, 0.3, length)
    pressure[change_at:] += 2.0
    temperature[change_at:] -= 1.2
    return ProcessRun(
        run_id="synthetic_relation_fault",
        timestamps=np.arange(length, dtype=float),
        values=np.column_stack((feed, pressure, temperature)),
        abnormal=np.arange(length) >= change_at,
        feature_names=("feed", "pressure", "temperature"),
        root_cause="pressure_relation_shift",
    )


def make_synthetic_causal_alarm_series(
    *, seed: int = 23, length: int = 1500, lag: int = 3
) -> dict[str, np.ndarray]:
    """Create binary alarms with a known ROOT -> TARGET delayed influence."""
    if length <= lag + 20 or lag < 1:
        raise ValueError("length must exceed lag and lag must be positive")
    rng = np.random.default_rng(seed)
    root = (rng.random(length) < 0.08).astype(np.int8)
    target = (rng.random(length) < 0.015).astype(np.int8)
    target[lag:] = np.maximum(
        target[lag:], root[:-lag] * (rng.random(length - lag) < 0.9)
    )
    distractor = (rng.random(length) < 0.09).astype(np.int8)
    return {"ROOT": root, "DISTRACTOR": distractor, "TARGET": target}
