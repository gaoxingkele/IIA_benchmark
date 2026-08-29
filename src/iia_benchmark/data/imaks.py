from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


IMAKS_SENSOR_MEMBER = "sensors/timeseries_annotated.csv"
IMAKS_NODE_MEMBER = "kg_seed/nodes.csv"
IMAKS_EDGE_MEMBER = "kg_seed/edges.csv"


@dataclass(frozen=True)
class IMAKSSensorData:
    """Aligned sensor values and annotations from the synthetic iMAKS plant."""

    timestamps: np.ndarray
    values: np.ndarray
    anomaly_labels: np.ndarray
    alarm_flags: np.ndarray
    sensor_names: tuple[str, ...]
    sample_seconds: float

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        values = np.asarray(self.values, dtype=float)
        labels = np.asarray(self.anomaly_labels, dtype=str)
        flags = np.asarray(self.alarm_flags, dtype=str)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "anomaly_labels", labels)
        object.__setattr__(self, "alarm_flags", flags)
        shape = (len(timestamps), len(self.sensor_names))
        if not len(timestamps) or values.shape != shape:
            raise ValueError("iMAKS values and sensor names do not align")
        if labels.shape != shape or flags.shape != shape:
            raise ValueError("iMAKS annotations do not align with sensor values")
        if not np.isfinite(values).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("iMAKS values must be finite and timestamps increasing")
        if self.sample_seconds <= 0 or len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("iMAKS sample period and sensor names are invalid")

    def series(self, name: str) -> np.ndarray:
        return self.values[:, self.sensor_names.index(name)].copy()

    def anomaly_state(self, name: str) -> np.ndarray:
        labels = self.anomaly_labels[:, self.sensor_names.index(name)]
        return (labels != "NORMAL").astype(np.int8)


@dataclass(frozen=True)
class IMAKSCausalEdge:
    source: str
    target: str
    relation: str
    rule_reference: str | None


def _pivot(frame: pd.DataFrame, column: str, timestamps: pd.DatetimeIndex, sensors: tuple[str, ...]) -> np.ndarray:
    values = frame.pivot(index="timestamp", columns="sensor_id", values=column)
    values = values.reindex(index=timestamps, columns=sensors)
    if values.isna().any().any():
        raise ValueError(f"iMAKS {column} matrix has missing sensor-time cells")
    return values.to_numpy()


def load_imaks_sensor_data(path: str | Path) -> IMAKSSensorData:
    """Load and align the annotated iMAKS long-form sensor table from ZIP."""

    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        if IMAKS_SENSOR_MEMBER not in archive.namelist():
            raise ValueError("iMAKS ZIP lacks the annotated sensor table")
        with archive.open(IMAKS_SENSOR_MEMBER) as stream:
            frame = pd.read_csv(stream, low_memory=False)
    required = {"timestamp", "sensor_id", "value", "anomaly_label", "alarm_flag"}
    if not required.issubset(frame.columns) or frame.duplicated(["timestamp", "sensor_id"]).any():
        raise ValueError("iMAKS sensor table has invalid columns or duplicate cells")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    timestamps = pd.DatetimeIndex(sorted(frame["timestamp"].unique()))
    sensors = tuple(sorted(map(str, frame["sensor_id"].unique())))
    seconds = timestamps.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    differences = np.diff(seconds)
    if not len(differences) or not np.allclose(differences, differences[0]):
        raise ValueError("iMAKS timestamp grid must be regular")
    return IMAKSSensorData(
        timestamps=seconds,
        values=_pivot(frame, "value", timestamps, sensors),
        anomaly_labels=_pivot(frame, "anomaly_label", timestamps, sensors),
        alarm_flags=_pivot(frame, "alarm_flag", timestamps, sensors),
        sensor_names=sensors,
        sample_seconds=float(differences[0]),
    )


def load_imaks_causal_edges(path: str | Path) -> tuple[IMAKSCausalEdge, ...]:
    """Resolve iMAKS KG edges to human-readable node names."""

    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        required = {IMAKS_NODE_MEMBER, IMAKS_EDGE_MEMBER}
        if not required.issubset(archive.namelist()):
            raise ValueError("iMAKS ZIP lacks KG nodes or edges")
        with archive.open(IMAKS_NODE_MEMBER) as stream:
            nodes = pd.read_csv(stream, low_memory=False)
        with archive.open(IMAKS_EDGE_MEMBER) as stream:
            edges = pd.read_csv(stream, low_memory=False)
    if not {"nodeId", "name"}.issubset(nodes.columns):
        raise ValueError("iMAKS node table has an invalid schema")
    if not {"fromId", "toId", "type", "ruleRef"}.issubset(edges.columns):
        raise ValueError("iMAKS edge table has an invalid schema")
    names = {
        str(row.nodeId): str(row.name) if pd.notna(row.name) else str(row.nodeId)
        for row in nodes.itertuples(index=False)
    }
    if any(str(value) not in names for value in (*edges["fromId"], *edges["toId"])):
        raise ValueError("iMAKS edge table references an unknown node")
    return tuple(
        IMAKSCausalEdge(
            source=names[str(row.fromId)],
            target=names[str(row.toId)],
            relation=str(row.type),
            rule_reference=None if pd.isna(row.ruleRef) else str(row.ruleRef),
        )
        for row in edges.itertuples(index=False)
    )
