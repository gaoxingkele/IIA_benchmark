from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ProcessRun


SKAB_LABEL_COLUMNS = frozenset({"datetime", "anomaly", "changepoint"})


def load_skab_csv(path: str | Path) -> ProcessRun:
    """Load one SKAB experiment without applying point adjustment.

    The anomaly-free training file has no label columns.  Labelled experiment
    files contain point-wise ``anomaly`` and ``changepoint`` columns; only the
    former is used as the detection target.  Keeping the original point labels
    prevents hidden post-processing from inflating validation scores.
    """

    source = Path(path)
    frame = pd.read_csv(source, sep=";")
    if "datetime" not in frame:
        raise ValueError("SKAB file must contain a datetime column")
    feature_names = tuple(
        column for column in frame.columns if column not in SKAB_LABEL_COLUMNS
    )
    if not feature_names:
        raise ValueError("SKAB file does not contain sensor features")
    values = frame.loc[:, feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("SKAB sensor values must be finite")
    timestamps = (
        pd.to_datetime(frame["datetime"], utc=True, errors="raise")
        .astype("int64")
        .to_numpy(dtype=float)
        / 1_000_000_000.0
    )
    if "anomaly" in frame:
        labels = frame["anomaly"].to_numpy(dtype=float)
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("SKAB anomaly labels must be binary")
        abnormal = labels.astype(bool)
    else:
        abnormal = np.zeros(len(frame), dtype=bool)
    return ProcessRun(
        run_id=source.stem,
        timestamps=timestamps,
        values=values,
        abnormal=abnormal,
        feature_names=feature_names,
    )
