from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .schema import AlarmEpisode, AlarmEvent


_FILE_NAME = re.compile(r"(?P<run>-?\d+)_alpha(?P<whole>\d+),(?P<fraction>\d+)\.csv")


@dataclass(frozen=True)
class NPPAlarmRun:
    fault_family: str
    run_number: int
    alpha: float
    timestamps: np.ndarray
    alarm_states: np.ndarray
    alarm_names: tuple[str, ...]
    source_samples: int
    sample_seconds: float = 10.0

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        raw_values = np.asarray(self.alarm_states)
        values = raw_values.astype(np.int8)
        if timestamps.ndim != 1 or values.ndim != 2 or len(timestamps) != len(values):
            raise ValueError("NPP timestamps and alarm states have incompatible dimensions")
        if len(timestamps) == 0 or values.shape[1] == 0:
            raise ValueError("NPP alarm run must be nonempty")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("NPP timestamps must be finite and strictly increasing")
        if not np.isfinite(raw_values).all() or not np.isin(raw_values, [0, 1]).all():
            raise ValueError("NPP alarm states must be binary")
        if len(self.alarm_names) != values.shape[1] or len(set(self.alarm_names)) != len(self.alarm_names):
            raise ValueError("NPP alarm names must be unique and match the state columns")
        if not self.fault_family or not np.isfinite(self.alpha):
            raise ValueError("NPP family or alpha is invalid")
        if self.source_samples < len(values) or self.sample_seconds <= 0:
            raise ValueError("NPP source length or sample period is invalid")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "alarm_states", values)

    @property
    def scenario(self) -> str:
        return self.fault_family

    @property
    def run_id(self) -> str:
        alpha = f"{self.alpha:.2f}".replace(".", "_")
        return f"{self.fault_family}_run{self.run_number:+d}_alpha{alpha}"

    @property
    def base_run_id(self) -> str:
        return f"{self.fault_family}_run{self.run_number:+d}"

    def representation(self, kind: str = "state") -> np.ndarray:
        if kind == "state":
            return self.alarm_states.copy()
        if kind != "rising_edge":
            raise ValueError("NPP representation must be 'state' or 'rising_edge'")
        previous = np.vstack(
            (
                np.zeros((1, self.alarm_states.shape[1]), dtype=np.int8),
                self.alarm_states[:-1],
            )
        )
        return np.maximum(self.alarm_states - previous, 0).astype(np.int8)

    def to_episode(self, *, include_clearances: bool = True) -> AlarmEpisode:
        previous = np.zeros(self.alarm_states.shape[1], dtype=np.int8)
        events: list[AlarmEvent] = []
        for sample, current in enumerate(self.alarm_states):
            for column in np.flatnonzero(current != previous):
                state = int(current[column])
                if state == 1 or include_clearances:
                    events.append(
                        AlarmEvent(
                            timestamp=float(self.timestamps[sample]),
                            tag=self.alarm_names[int(column)],
                            state=state,
                        )
                    )
            previous = current
        return AlarmEpisode(
            episode_id=self.run_id,
            events=tuple(events),
            label=self.fault_family,
            root_cause=None if self.fault_family == "Normal" else self.fault_family,
        )


@dataclass(frozen=True)
class NPPAlarmSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calibration: np.ndarray
    y_calibration: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    alarm_names: tuple[str, ...]
    train_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    test_run_ids: tuple[str, ...]
    unused_run_ids: tuple[str, ...]
    conflicting_run_ids: tuple[str, ...]
    duplicate_run_ids: tuple[str, ...]
    representation: str
    random_state: int


def load_npp_alarm_runs(
    path: str | Path,
    *,
    alpha: float | None = None,
    fault_families: tuple[str, ...] | list[str] | None = None,
    minimum_samples: int = 1,
    horizon_samples: int | None = None,
    include_normal: bool = False,
) -> tuple[NPPAlarmRun, ...]:
    """Load one extracted NPP alpha slice with strict schema and time checks."""

    root = Path(path)
    allowed = set(fault_families) if fault_families is not None else None
    if minimum_samples <= 0 or (horizon_samples is not None and horizon_samples <= 0):
        raise ValueError("NPP sample limits must be positive")
    records: list[NPPAlarmRun] = []
    expected_names: tuple[str, ...] | None = None
    observed_ids: set[tuple[str, int, float]] = set()
    for source in sorted(root.rglob("*.csv")):
        match = _FILE_NAME.fullmatch(source.name)
        if match is None:
            continue
        family = source.parent.name
        if family == "Normal" and not include_normal:
            continue
        if allowed is not None and family not in allowed:
            continue
        file_alpha = float(f"{match.group('whole')}.{match.group('fraction')}")
        if alpha is not None and not np.isclose(file_alpha, alpha, atol=1e-12):
            continue
        run_number = int(match.group("run"))
        identity = (family, run_number, file_alpha)
        if identity in observed_ids:
            raise ValueError(f"duplicate NPP run identity: {identity}")
        observed_ids.add(identity)
        frame = pd.read_csv(source)
        if not len(frame) or str(frame.columns[0]) != "TIME":
            raise ValueError(f"NPP CSV must start with TIME: {source}")
        source_samples = len(frame)
        if source_samples < minimum_samples:
            continue
        stop = source_samples if horizon_samples is None else min(source_samples, horizon_samples)
        alarm_frame = frame.iloc[:stop, 1:]
        names = tuple(map(str, alarm_frame.columns))
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f"NPP alarm columns differ: {source}")
        timestamps = frame.iloc[:stop, 0].to_numpy(dtype=float)
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 10.0):
            raise ValueError(f"NPP time grid is not 10 seconds: {source}")
        records.append(
            NPPAlarmRun(
                fault_family=family,
                run_number=run_number,
                alpha=file_alpha,
                timestamps=timestamps,
                alarm_states=alarm_frame.to_numpy(),
                alarm_names=names,
                source_samples=source_samples,
            )
        )
    if not records:
        raise ValueError("NPP alarm selection produced no runs")
    if horizon_samples is not None and any(len(run.timestamps) != horizon_samples for run in records):
        raise ValueError("NPP minimum_samples must cover the registered fixed horizon")
    return tuple(sorted(records, key=lambda run: (run.fault_family, run.run_number)))


def build_npp_alarm_split(
    runs: tuple[NPPAlarmRun, ...] | list[NPPAlarmRun],
    *,
    train_per_class: int,
    calibration_per_class: int,
    test_per_class: int,
    random_state: int,
    representation: str = "state",
) -> NPPAlarmSplit:
    """Seeded balanced split grouped by exact state-or-edge trajectory components."""

    if not runs:
        raise ValueError("NPP split requires runs")
    counts = (int(train_per_class), int(calibration_per_class), int(test_per_class))
    if any(count <= 0 for count in counts):
        raise ValueError("NPP split counts must be positive")
    alarm_names = runs[0].alarm_names
    shape = runs[0].alarm_states.shape
    if any(run.alarm_names != alarm_names or run.alarm_states.shape != shape for run in runs):
        raise ValueError("NPP runs must share alarm columns and fixed horizon")
    if len({run.alpha for run in runs}) != 1:
        raise ValueError("NPP base split must contain exactly one alpha slice")
    rng = np.random.default_rng(int(random_state))
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for kind in ("state", "rising_edge"):
        seen: dict[bytes, int] = {}
        for index, run in enumerate(runs):
            signature = run.representation(kind).tobytes()
            if signature in seen:
                union(index, seen[signature])
            else:
                seen[signature] = index
    components_by_root: dict[int, list[NPPAlarmRun]] = {}
    for index, run in enumerate(runs):
        components_by_root.setdefault(find(index), []).append(run)
    conflicting_components = [
        component
        for component in components_by_root.values()
        if len({run.fault_family for run in component}) > 1
    ]
    valid_components = [
        sorted(component, key=lambda run: run.run_number)
        for component in components_by_root.values()
        if len({run.fault_family for run in component}) == 1
    ]
    partitions: dict[str, list[NPPAlarmRun]] = {
        "train": [],
        "calibration": [],
        "test": [],
        "unused": [],
    }
    required = sum(counts)
    duplicate_run_ids: list[str] = []
    for label in sorted({run.fault_family for run in runs}):
        class_components = sorted(
            (
                component
                for component in valid_components
                if component[0].fault_family == label
            ),
            key=lambda component: component[0].run_number,
        )
        if len(class_components) < required:
            raise ValueError(
                f"NPP class {label} has {len(class_components)} independent trajectory groups; {required} required"
            )
        shuffled_components = [
            class_components[index] for index in rng.permutation(len(class_components))
        ]
        representatives: list[NPPAlarmRun] = []
        for component in shuffled_components:
            representative_index = int(rng.integers(0, len(component)))
            representatives.append(component[representative_index])
            duplicate_run_ids.extend(
                run.run_id
                for index, run in enumerate(component)
                if index != representative_index
            )
        train_stop = counts[0]
        calibration_stop = train_stop + counts[1]
        test_stop = calibration_stop + counts[2]
        partitions["train"].extend(representatives[:train_stop])
        partitions["calibration"].extend(representatives[train_stop:calibration_stop])
        partitions["test"].extend(representatives[calibration_stop:test_stop])
        partitions["unused"].extend(representatives[test_stop:])

    def arrays(name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        selected = partitions[name]
        return (
            np.stack([run.representation(representation).T for run in selected]),
            np.asarray([run.fault_family for run in selected]),
            tuple(run.run_id for run in selected),
        )

    X_train, y_train, train_ids = arrays("train")
    X_calibration, y_calibration, calibration_ids = arrays("calibration")
    X_test, y_test, test_ids = arrays("test")
    used = set(train_ids) | set(calibration_ids) | set(test_ids)
    if len(used) != len(train_ids) + len(calibration_ids) + len(test_ids):
        raise RuntimeError("NPP split unexpectedly overlaps run identities")
    return NPPAlarmSplit(
        X_train=X_train,
        y_train=y_train,
        X_calibration=X_calibration,
        y_calibration=y_calibration,
        X_test=X_test,
        y_test=y_test,
        alarm_names=alarm_names,
        train_run_ids=train_ids,
        calibration_run_ids=calibration_ids,
        test_run_ids=test_ids,
        unused_run_ids=tuple(
            run.run_id for run in partitions["unused"]
        )
        + tuple(duplicate_run_ids)
        + tuple(
            run.run_id
            for component in conflicting_components
            for run in component
        ),
        conflicting_run_ids=tuple(
            run.run_id
            for component in conflicting_components
            for run in component
        ),
        duplicate_run_ids=tuple(duplicate_run_ids),
        representation=representation,
        random_state=int(random_state),
    )
