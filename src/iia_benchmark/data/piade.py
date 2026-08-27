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
