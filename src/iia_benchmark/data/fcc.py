from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from .schema import AlarmEpisode, AlarmEvent


_ALARM_MEMBER = re.compile(
    r"^alarmseriesdata/(?P<label>[^/]+)/(?P=label)_run(?P<run>\d+)_alarm\.csv$"
)
_SERIES_MEMBER = re.compile(
    r"^timeseriesdata/(?P<label>[^/]+)/(?P=label)_run(?P<run>\d+)_(?P<kind>process|valves|disturbances)\.csv$"
)


@dataclass(frozen=True)
class FCCAlarmRun:
    scenario: str
    run_number: int
    alarm_states: np.ndarray
    alarm_names: tuple[str, ...]
    sample_minutes: float = 1.0

    def __post_init__(self) -> None:
        raw_values = np.asarray(self.alarm_states)
        values = raw_values.astype(np.int8)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("FCC alarm states must be a nonempty matrix")
        if not np.isfinite(raw_values).all() or not np.isin(raw_values, [0, 1]).all():
            raise ValueError("FCC alarm states must be binary")
        object.__setattr__(self, "alarm_states", values)
        if len(self.alarm_names) != values.shape[1]:
            raise ValueError("FCC alarm names must match state columns")
        if not self.scenario or self.run_number < 1 or self.sample_minutes <= 0:
            raise ValueError("FCC scenario, run number, and sample period are invalid")

    @property
    def run_id(self) -> str:
        return f"{self.scenario}_run{self.run_number}"

    def representation(self, kind: str = "state") -> np.ndarray:
        if kind == "state":
            return self.alarm_states.copy()
        if kind != "rising_edge":
            raise ValueError("FCC representation must be 'state' or 'rising_edge'")
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
            changed = np.flatnonzero(current != previous)
            for column in changed:
                state = int(current[column])
                if state == 1 or include_clearances:
                    events.append(
                        AlarmEvent(
                            timestamp=float(sample) * self.sample_minutes * 60.0,
                            tag=self.alarm_names[int(column)],
                            state=state,
                        )
                    )
            previous = current
        return AlarmEpisode(
            episode_id=self.run_id,
            events=tuple(events),
            label=self.scenario,
            root_cause=self.scenario,
        )


@dataclass(frozen=True)
class FCCTimeSeriesRun:
    scenario: str
    run_number: int
    timestamps: np.ndarray
    process_values: np.ndarray
    valve_values: np.ndarray
    disturbance_values: np.ndarray
    process_names: tuple[str, ...]
    valve_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        process = np.asarray(self.process_values, dtype=float)
        valves = np.asarray(self.valve_values, dtype=float)
        disturbances = np.asarray(self.disturbance_values, dtype=float)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "process_values", process)
        object.__setattr__(self, "valve_values", valves)
        object.__setattr__(self, "disturbance_values", disturbances)
        if timestamps.ndim != 1 or not len(timestamps):
            raise ValueError("FCC timestamps must be a nonempty vector")
        for name, values, columns in (
            ("process", process, self.process_names),
            ("valve", valves, self.valve_names),
            ("disturbance", disturbances, self.disturbance_names),
        ):
            if values.ndim != 2 or len(values) != len(timestamps):
                raise ValueError(f"FCC {name} values have invalid dimensions")
            if values.shape[1] != len(columns) or np.isinf(values).any():
                raise ValueError(f"FCC {name} values, infinities, or names are invalid")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("FCC timestamps must be strictly increasing")

    @property
    def run_id(self) -> str:
        return f"{self.scenario}_run{self.run_number}"

    @property
    def root_disturbance(self) -> str:
        mapping = {
            "catalyst_deactivation": "Catalyst_Deactivation",
            "cyclone_damage": "Cyclone_Damage",
            "preheater_shutdown": "Preheater_Shutdown",
            "preheater_temp_increase": "Preheater_Increase",
        }
        if self.scenario in mapping:
            return mapping[self.scenario]
        valve = self.scenario.split("_", 1)[0]
        return f"Dist_{valve}"


@dataclass(frozen=True)
class FCCAlarmSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calibration: np.ndarray
    y_calibration: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    alarm_names: tuple[str, ...]
    train_run_numbers: tuple[int, ...]
    calibration_run_numbers: tuple[int, ...]
    test_run_numbers: tuple[int, ...]
    representation: str


def _selected(value: str | int, allowed: set[str] | set[int] | None) -> bool:
    return allowed is None or value in allowed


def load_fcc_alarm_runs(
    path: str | Path,
    *,
    scenarios: tuple[str, ...] | list[str] | None = None,
    run_numbers: tuple[int, ...] | list[int] | None = None,
) -> tuple[FCCAlarmRun, ...]:
    """Load run-labelled FCC binary alarm matrices directly from the official ZIP."""

    source = Path(path)
    scenario_filter = set(scenarios) if scenarios is not None else None
    run_filter = set(map(int, run_numbers)) if run_numbers is not None else None
    records: list[FCCAlarmRun] = []
    expected_names: tuple[str, ...] | None = None
    with zipfile.ZipFile(source) as archive:
        for member in sorted(archive.namelist()):
            match = _ALARM_MEMBER.fullmatch(member.replace("\\", "/"))
            if match is None:
                continue
            scenario = match.group("label")
            run_number = int(match.group("run"))
            if not _selected(scenario, scenario_filter) or not _selected(
                run_number, run_filter
            ):
                continue
            with archive.open(member) as stream:
                frame = pd.read_csv(stream)
            names = tuple(map(str, frame.columns))
            if frame.columns.duplicated().any():
                raise ValueError(f"FCC alarm file has duplicate columns: {member}")
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ValueError(f"FCC alarm columns differ: {member}")
            records.append(
                FCCAlarmRun(
                    scenario=scenario,
                    run_number=run_number,
                    alarm_states=frame.to_numpy(),
                    alarm_names=names,
                )
            )
    if not records:
        raise ValueError("FCC alarm selection produced no runs")
    return tuple(sorted(records, key=lambda run: (run.scenario, run.run_number)))


def load_fcc_timeseries_runs(
    path: str | Path,
    *,
    scenarios: tuple[str, ...] | list[str] | None = None,
    run_numbers: tuple[int, ...] | list[int] | None = None,
) -> tuple[FCCTimeSeriesRun, ...]:
    """Load aligned FCC process, valve, and injected-disturbance trajectories."""

    source = Path(path)
    scenario_filter = set(scenarios) if scenarios is not None else None
    run_filter = set(map(int, run_numbers)) if run_numbers is not None else None
    members: dict[tuple[str, int], dict[str, str]] = {}
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            match = _SERIES_MEMBER.fullmatch(member.replace("\\", "/"))
            if match is None:
                continue
            scenario = match.group("label")
            run_number = int(match.group("run"))
            if not _selected(scenario, scenario_filter) or not _selected(
                run_number, run_filter
            ):
                continue
            members.setdefault((scenario, run_number), {})[match.group("kind")] = member
        records: list[FCCTimeSeriesRun] = []
        expected_columns: dict[str, tuple[str, ...]] = {}
        for (scenario, run_number), by_kind in sorted(members.items()):
            if set(by_kind) != {"process", "valves", "disturbances"}:
                raise ValueError(f"incomplete FCC timeseries triplet: {scenario} run {run_number}")
            frames: dict[str, pd.DataFrame] = {}
            for kind, member in by_kind.items():
                with archive.open(member) as stream:
                    frames[kind] = pd.read_csv(stream)
                columns = tuple(map(str, frames[kind].columns))
                if not columns or columns[0] != "Time":
                    raise ValueError(f"FCC {kind} table must start with Time: {member}")
                if kind not in expected_columns:
                    expected_columns[kind] = columns
                elif columns != expected_columns[kind]:
                    raise ValueError(f"FCC {kind} columns differ: {member}")
            timestamps = frames["process"]["Time"].to_numpy(dtype=float)
            for kind in ("valves", "disturbances"):
                if not np.array_equal(
                    timestamps, frames[kind]["Time"].to_numpy(dtype=float)
                ):
                    raise ValueError(
                        f"FCC {kind} timestamps are not aligned: {scenario} run {run_number}"
                    )
            records.append(
                FCCTimeSeriesRun(
                    scenario=scenario,
                    run_number=run_number,
                    timestamps=timestamps,
                    process_values=frames["process"].iloc[:, 1:].to_numpy(),
                    valve_values=frames["valves"].iloc[:, 1:].to_numpy(),
                    disturbance_values=frames["disturbances"].iloc[:, 1:].to_numpy(),
                    process_names=tuple(map(str, frames["process"].columns[1:])),
                    valve_names=tuple(map(str, frames["valves"].columns[1:])),
                    disturbance_names=tuple(
                        map(str, frames["disturbances"].columns[1:])
                    ),
                )
            )
    if not records:
        raise ValueError("FCC timeseries selection produced no runs")
    return tuple(records)


def build_fcc_alarm_split(
    runs: tuple[FCCAlarmRun, ...] | list[FCCAlarmRun],
    *,
    train_run_numbers: tuple[int, ...] | list[int],
    calibration_run_numbers: tuple[int, ...] | list[int],
    test_run_numbers: tuple[int, ...] | list[int],
    representation: str = "state",
) -> FCCAlarmSplit:
    """Create a strict, complete-run FCC train/calibration/test split."""

    if not runs:
        raise ValueError("FCC split requires runs")
    train_ids = tuple(map(int, train_run_numbers))
    calibration_ids = tuple(map(int, calibration_run_numbers))
    test_ids = tuple(map(int, test_run_numbers))
    id_sets = [set(train_ids), set(calibration_ids), set(test_ids)]
    if any(not values for values in id_sets):
        raise ValueError("FCC split partitions must be nonempty")
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise ValueError("FCC split run numbers must be disjoint")
    alarm_names = runs[0].alarm_names
    if any(run.alarm_names != alarm_names for run in runs):
        raise ValueError("FCC runs must share alarm columns")
    if any(run.alarm_states.shape != runs[0].alarm_states.shape for run in runs):
        raise ValueError("FCC runs must share matrix dimensions")

    def select(numbers: set[int]) -> tuple[np.ndarray, np.ndarray]:
        chosen = [run for run in runs if run.run_number in numbers]
        labels = {run.scenario for run in runs}
        if {run.scenario for run in chosen} != labels:
            raise ValueError("every FCC split partition must cover every scenario")
        X = np.stack([run.representation(representation).T for run in chosen])
        y = np.asarray([run.scenario for run in chosen])
        return X, y

    X_train, y_train = select(id_sets[0])
    X_calibration, y_calibration = select(id_sets[1])
    X_test, y_test = select(id_sets[2])
    return FCCAlarmSplit(
        X_train=X_train,
        y_train=y_train,
        X_calibration=X_calibration,
        y_calibration=y_calibration,
        X_test=X_test,
        y_test=y_test,
        alarm_names=alarm_names,
        train_run_numbers=train_ids,
        calibration_run_numbers=calibration_ids,
        test_run_numbers=test_ids,
        representation=representation,
    )
