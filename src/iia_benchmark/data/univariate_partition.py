"""Configuration-driven, read-only loading for a univariate transfer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


PARTITIONS = (
    "normal_train",
    "normal_evaluation",
    "abnormal_calibration",
    "abnormal_evaluation",
)


@dataclass(frozen=True)
class UnivariateTransferBundle:
    dataset_id: str
    feature_name: str
    sample_period_seconds: float
    normal_train: np.ndarray
    normal_evaluation: np.ndarray
    abnormal_calibration: np.ndarray
    abnormal_evaluation: np.ndarray
    partition_sources: dict[str, dict[str, object]]
    leaderboard_eligible: bool
    citation: dict[str, str]


def _filtered_partition(
    root: Path, name: str, specification: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    if specification.get("loader") != "csv":
        raise ValueError(f"{name}.loader must be 'csv'")
    path = (root / str(specification["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    delimiter = str(specification.get("delimiter", ","))
    frame = pd.read_csv(path, sep=delimiter)
    filters = specification.get("filters", {})
    if not isinstance(filters, dict):
        raise ValueError(f"{name}.filters must be an object")
    for column, expected in filters.items():
        if column not in frame.columns:
            raise ValueError(f"{name} filter column {column!r} is missing")
        frame = frame.loc[frame[column] == expected]
    start = int(specification.get("row_start", 0))
    stop_value = specification.get("row_stop")
    stop = len(frame) if stop_value is None else int(stop_value)
    if start < 0 or stop <= start or stop > len(frame):
        raise ValueError(f"{name} has an invalid filtered row interval [{start}, {stop})")
    column = str(specification["value_column"])
    if column not in frame.columns:
        raise ValueError(f"{name} value column {column!r} is missing")
    values = frame.iloc[start:stop][column].to_numpy(dtype=float, copy=True)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite values")
    return values, {
        "path": path.as_posix(),
        "group_id": str(specification["group_id"]),
        "filters": filters,
        "delimiter": delimiter,
        "filtered_row_start": start,
        "filtered_row_stop": stop,
        "samples": len(values),
    }


def _check_intervals(sources: dict[str, dict[str, object]]) -> None:
    names = tuple(sources)
    for left_index, left_name in enumerate(names):
        left = sources[left_name]
        for right_name in names[left_index + 1 :]:
            right = sources[right_name]
            if left["path"] != right["path"] or left["filters"] != right["filters"]:
                continue
            overlap = max(
                int(left["filtered_row_start"]), int(right["filtered_row_start"])
            ) < min(int(left["filtered_row_stop"]), int(right["filtered_row_stop"]))
            if overlap:
                raise ValueError(
                    f"partition overlap between {left_name} and {right_name}"
                )


def load_univariate_transfer_config(
    config_path: str | Path, *, root: str | Path | None = None
) -> tuple[dict[str, object], UnivariateTransferBundle]:
    """Load four registered partitions without modifying their source files."""

    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "id",
        "feature_name",
        "sample_period_seconds",
        "leaderboard_eligible",
        "citation",
        "partitions",
        "adaptation",
        "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    sample_period = float(config["sample_period_seconds"])
    if sample_period <= 0:
        raise ValueError("sample_period_seconds must be positive")
    partitions = config["partitions"]
    if set(partitions) != set(PARTITIONS):
        raise ValueError(f"partitions must be exactly {list(PARTITIONS)}")
    data_root = Path(root).resolve() if root is not None else path.parent
    arrays = {}
    sources = {}
    for name in PARTITIONS:
        arrays[name], sources[name] = _filtered_partition(
            data_root, name, partitions[name]
        )
    _check_intervals(sources)
    if bool(config["leaderboard_eligible"]):
        groups = [sources[name]["group_id"] for name in PARTITIONS]
        if len(set(groups)) != len(groups):
            raise ValueError(
                "leaderboard-eligible partitions must have distinct group_id values"
            )
    citation = config["citation"]
    if not isinstance(citation, dict) or not citation.get("title") or not (
        citation.get("doi") or citation.get("url")
    ):
        raise ValueError("citation requires title and either doi or url")
    return config, UnivariateTransferBundle(
        dataset_id=str(config["id"]),
        feature_name=str(config["feature_name"]),
        sample_period_seconds=sample_period,
        normal_train=arrays["normal_train"],
        normal_evaluation=arrays["normal_evaluation"],
        abnormal_calibration=arrays["abnormal_calibration"],
        abnormal_evaluation=arrays["abnormal_evaluation"],
        partition_sources=sources,
        leaderboard_eligible=bool(config["leaderboard_eligible"]),
        citation={str(key): str(value) for key, value in citation.items()},
    )
