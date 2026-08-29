from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ENAS_ERROR_NAMES = ("ME", "HE", "UE")
ENAS_METADATA_COLUMNS = frozenset({"Timestamp", "ME", "HE", "UE", "PV"})


@dataclass(frozen=True)
class EnASEventLog:
    """Exception-logged EnAS digital states and manual error annotations."""

    timestamps: np.ndarray
    signal_states: np.ndarray
    error_states: np.ndarray
    production_variant: np.ndarray
    signal_names: tuple[str, ...]
    error_names: tuple[str, ...] = ENAS_ERROR_NAMES

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        signals = np.asarray(self.signal_states, dtype=np.int8)
        errors = np.asarray(self.error_states, dtype=np.int8)
        variants = np.asarray(self.production_variant, dtype=np.int8)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "signal_states", signals)
        object.__setattr__(self, "error_states", errors)
        object.__setattr__(self, "production_variant", variants)
        if timestamps.ndim != 1 or not len(timestamps):
            raise ValueError("EnAS timestamps must be a nonempty vector")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("EnAS timestamps must be strictly increasing")
        if signals.shape != (len(timestamps), len(self.signal_names)):
            raise ValueError("EnAS signal states and names do not align")
        if errors.shape != (len(timestamps), len(self.error_names)):
            raise ValueError("EnAS error states and names do not align")
        if variants.shape != (len(timestamps),):
            raise ValueError("EnAS production variants do not align")
        if not np.isin(signals, (0, 1)).all() or not np.isin(errors, (0, 1)).all():
            raise ValueError("EnAS signals and error markers must be binary")
        if not np.isin(variants, (1, 2)).all():
            raise ValueError("EnAS production variants must be 1 or 2")
        if len(set(self.signal_names)) != len(self.signal_names):
            raise ValueError("EnAS signal names must be unique")

    def signal(self, name: str) -> np.ndarray:
        return self.signal_states[:, self.signal_names.index(name)].copy()

    def error(self, name: str) -> np.ndarray:
        return self.error_states[:, self.error_names.index(name)].copy()


def load_enas_event_log(path: str | Path) -> EnASEventLog:
    """Load the public EnAS event log without inventing a polling grid.

    Every source row is retained because the dataset is logged by exception.
    ME/HE/UE remain instantaneous manual annotations; callers must explicitly
    declare any persistence window used by an online alarm algorithm.
    """

    source = Path(path)
    frame = pd.read_csv(source)
    required = {"Timestamp", "ME", "HE", "UE", "PV"}
    if not required.issubset(frame.columns) or frame.columns.duplicated().any():
        raise ValueError("EnAS table is missing required or has duplicate columns")
    signal_names = tuple(
        str(column) for column in frame.columns if column not in ENAS_METADATA_COLUMNS
    )
    if not signal_names:
        raise ValueError("EnAS table has no digital signals")
    parsed = pd.to_datetime(frame["Timestamp"], utc=True, errors="raise")
    timestamps = parsed.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    return EnASEventLog(
        timestamps=timestamps,
        signal_states=frame.loc[:, signal_names].to_numpy(),
        error_states=frame.loc[:, ENAS_ERROR_NAMES].to_numpy(),
        production_variant=frame["PV"].to_numpy(),
        signal_names=signal_names,
    )
