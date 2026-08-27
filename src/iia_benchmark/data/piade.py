from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import AlarmEvent


def load_piade_alarm_events(
    path: str | Path,
    *,
    equipment_id: str | None = None,
    exclude_codes: tuple[str, ...] = ("A_000",),
) -> tuple[AlarmEvent, ...]:
    """Load activations from the public PIADE ``raw_data.csv`` interval table."""
    frame = pd.read_csv(path, usecols=["equipment_ID", "alarm", "start"])
    if equipment_id is not None:
        frame = frame.loc[frame["equipment_ID"] == equipment_id]
    frame = frame.dropna(subset=["alarm", "start"])
    frame = frame.loc[~frame["alarm"].astype(str).isin(exclude_codes)]
    events = (
        AlarmEvent(timestamp=float(row.start), tag=str(row.alarm))
        for row in frame.itertuples(index=False)
    )
    return tuple(sorted(events, key=lambda event: event.timestamp))


def load_piade_alarm_intervals(
    path: str | Path,
    *,
    equipment_id: str | None = None,
    exclude_codes: tuple[str, ...] = ("A_000",),
) -> tuple[AlarmEvent, ...]:
    """Load PIADE alarm intervals as explicit activation/clearance events."""

    frame = pd.read_csv(path, usecols=["equipment_ID", "alarm", "start", "end"])
    if equipment_id is not None:
        frame = frame.loc[frame["equipment_ID"] == equipment_id]
    frame = frame.dropna(subset=["alarm", "start"])
    frame = frame.loc[~frame["alarm"].astype(str).isin(exclude_codes)]
    events: list[AlarmEvent] = []
    for row in frame.itertuples(index=False):
        tag = str(row.alarm)
        events.append(AlarmEvent(timestamp=float(row.start), tag=tag, state=1))
        if pd.notna(row.end) and float(row.end) >= float(row.start):
            events.append(AlarmEvent(timestamp=float(row.end), tag=tag, state=0))
    return tuple(sorted(events, key=lambda event: (event.timestamp, event.tag, event.state)))


def load_piade_alarm_sequences(
    path: str | Path,
    *,
    window_seconds: float = 86_400.0,
    exclude_codes: tuple[str, ...] = ("A_000",),
) -> dict[str, tuple[tuple[AlarmEvent, ...], ...]]:
    """Group PIADE activations into fixed chronological windows per equipment."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    frame = pd.read_csv(path, usecols=["equipment_ID", "alarm", "start"])
    frame = frame.dropna(subset=["equipment_ID", "alarm", "start"])
    frame = frame.loc[~frame["alarm"].astype(str).isin(exclude_codes)]
    grouped: dict[str, tuple[tuple[AlarmEvent, ...], ...]] = {}
    for equipment, equipment_frame in frame.groupby("equipment_ID", sort=True):
        equipment_frame = equipment_frame.sort_values("start", kind="stable")
        origin = float(equipment_frame["start"].min())
        bins = ((equipment_frame["start"].astype(float) - origin) // window_seconds).astype(int)
        sequences = []
        for _, window in equipment_frame.groupby(bins, sort=True):
            events = tuple(
                AlarmEvent(timestamp=float(row.start), tag=str(row.alarm))
                for row in window.itertuples(index=False)
            )
            if len(events) >= 2:
                sequences.append(events)
        grouped[str(equipment)] = tuple(sequences)
    return grouped
