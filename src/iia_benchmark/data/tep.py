from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import ProcessRun


TEP_FEATURE_NAMES = tuple(
    [f"XMEAS_{index:02d}" for index in range(1, 42)]
    + [f"XMV_{index:02d}" for index in range(1, 12)]
)


def load_tep_ascii(
    path: str | Path,
    *,
    fault_start: int | None = None,
    root_cause: str | None = None,
    sample_period: float = 1.0,
) -> ProcessRun:
    """Load the classic Braatz TEP whitespace format into a ``ProcessRun``.

    The distributed files are often stored as 52 variable rows by N samples,
    despite the accompanying logical N-by-52 description. Orientation is
    detected explicitly. ``fault_start`` is caller supplied because injection
    conventions differ between train/test releases.
    """
    matrix = np.loadtxt(path, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TEP file must contain a numeric matrix")
    if matrix.shape[1] == 52:
        values = matrix
    elif matrix.shape[0] == 52:
        values = matrix.T
    else:
        raise ValueError(f"expected one TEP dimension to equal 52, got {matrix.shape}")
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    abnormal = np.zeros(len(values), dtype=bool)
    if fault_start is not None:
        if not 0 <= fault_start < len(values):
            raise ValueError("fault_start must index the loaded run")
        abnormal[fault_start:] = True
    return ProcessRun(
        run_id=Path(path).stem,
        timestamps=np.arange(len(values), dtype=float) * sample_period,
        values=values,
        abnormal=abnormal,
        feature_names=TEP_FEATURE_NAMES,
        root_cause=root_cause,
    )
