from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, order=True)
class AlarmEvent:
    timestamp: float
    tag: str
    state: int = 1
    priority: int = 1

    def __post_init__(self) -> None:
        if self.state not in (0, 1):
            raise ValueError("AlarmEvent.state must be 0 (clearance) or 1 (activation)")


@dataclass(frozen=True)
class AlarmEpisode:
    episode_id: str
    events: tuple[AlarmEvent, ...]
    label: str | None = None
    root_cause: str | None = None

    def activations(self) -> tuple[AlarmEvent, ...]:
        return tuple(event for event in self.events if event.state == 1)

    def tags(self, *, unique: bool = False) -> tuple[str, ...]:
        tags = tuple(event.tag for event in self.activations())
        return tuple(dict.fromkeys(tags)) if unique else tags


@dataclass(frozen=True)
class ProcessRun:
    run_id: str
    timestamps: np.ndarray
    values: np.ndarray
    abnormal: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    root_cause: str | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        values = np.asarray(self.values, dtype=float)
        abnormal = np.asarray(self.abnormal, dtype=bool)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "abnormal", abnormal)
        if timestamps.ndim != 1 or abnormal.ndim != 1:
            raise ValueError("timestamps and abnormal must be one-dimensional")
        if values.ndim == 1:
            values = values[:, None]
            object.__setattr__(self, "values", values)
        if values.ndim != 2:
            raise ValueError("values must be a one- or two-dimensional array")
        if not (len(timestamps) == len(values) == len(abnormal)):
            raise ValueError("ProcessRun arrays must have equal first dimensions")
        if self.feature_names and len(self.feature_names) != values.shape[1]:
            raise ValueError("feature_names must match the number of value columns")


def alarm_events_to_state_matrix(
    events: tuple[AlarmEvent, ...] | list[AlarmEvent],
    *,
    sample_seconds: float = 1.0,
    start: float | None = None,
    stop: float | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Replay activation/clearance events onto an inclusive fixed-time grid."""

    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    ordered = tuple(sorted(events, key=lambda event: event.timestamp))
    if not ordered:
        raise ValueError("at least one alarm event is required")
    first = ordered[0].timestamp if start is None else float(start)
    last = ordered[-1].timestamp if stop is None else float(stop)
    if last < first:
        raise ValueError("stop must not precede start")
    names = tuple(sorted({event.tag for event in ordered}))
    name_to_column = {name: index for index, name in enumerate(names)}
    grid = np.arange(first, last + sample_seconds, sample_seconds, dtype=float)
    states = np.zeros((len(grid), len(names)), dtype=np.int8)
    current = np.zeros(len(names), dtype=np.int8)
    event_index = 0
    for row, timestamp in enumerate(grid):
        while event_index < len(ordered) and ordered[event_index].timestamp <= timestamp:
            event = ordered[event_index]
            current[name_to_column[event.tag]] = event.state
            event_index += 1
        states[row] = current
    return grid, names, states
