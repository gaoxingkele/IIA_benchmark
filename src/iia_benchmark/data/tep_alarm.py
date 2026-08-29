from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from .schema import AlarmEpisode, AlarmEvent


_CSV_MEMBER = re.compile(
    r"^.*/class_(?P<class_label>\d+)/alarm_timeseries_(?P<sample>\d+)\.csv$"
)
_GROUND_TRUTH_COLUMNS = (
    "sample ID",
    "class label",
    "disturbance name",
    "min. scaling",
    "max. scaling",
)


@dataclass(frozen=True)
class TEPAlarmRun:
    disturbance: str
    class_label: int
    sample_number: int
    class_position: int
    min_scaling: float
    max_scaling: float
    alarm_states: np.ndarray
    alarm_names: tuple[str, ...]
    sample_minutes: float = 1.0

    def __post_init__(self) -> None:
        raw_values = np.asarray(self.alarm_states)
        values = raw_values.astype(np.int8)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("TEP alarm states must be a nonempty matrix")
        if not np.isfinite(raw_values).all() or not np.isin(raw_values, [0, 1]).all():
            raise ValueError("TEP alarm states must be binary")
        object.__setattr__(self, "alarm_states", values)
        if len(self.alarm_names) != values.shape[1]:
            raise ValueError("TEP alarm names must match state columns")
        if not self.disturbance or self.class_label < 0:
            raise ValueError("TEP class metadata is invalid")
        if self.sample_number < 1 or self.class_position < 1:
            raise ValueError("TEP sample identifiers must be positive")
        if not np.isfinite([self.min_scaling, self.max_scaling]).all():
            raise ValueError("TEP scaling metadata must be finite")
        if self.min_scaling > self.max_scaling or self.sample_minutes <= 0:
            raise ValueError("TEP scaling range or sample period is invalid")

    @property
    def scenario(self) -> str:
        return self.disturbance

    @property
    def run_number(self) -> int:
        return self.sample_number

    @property
    def run_id(self) -> str:
        return f"class_{self.class_label}_sample_{self.sample_number}"

    def representation(self, kind: str = "state") -> np.ndarray:
        if kind == "state":
            return self.alarm_states.copy()
        if kind != "rising_edge":
            raise ValueError("TEP representation must be 'state' or 'rising_edge'")
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
                            timestamp=float(sample) * self.sample_minutes * 60.0,
                            tag=self.alarm_names[int(column)],
                            state=state,
                        )
                    )
            previous = current
        return AlarmEpisode(
            episode_id=self.run_id,
            events=tuple(events),
            label=self.disturbance,
            root_cause=self.disturbance,
        )


@dataclass(frozen=True)
class TEPAlarmSplit:
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
    representation: str
    random_state: int


def _ground_truth(archive: zipfile.ZipFile) -> pd.DataFrame:
    members = [
        name for name in archive.namelist() if name.lower().endswith("ground_truth.xlsx")
    ]
    if len(members) != 1:
        raise ValueError("TEP archive must contain exactly one ground_truth.xlsx")
    frame = pd.read_excel(BytesIO(archive.read(members[0])))
    if tuple(map(str, frame.columns)) != _GROUND_TRUTH_COLUMNS:
        raise ValueError("TEP ground-truth columns do not match the registered schema")
    if frame.isna().any().any() or frame["sample ID"].duplicated().any():
        raise ValueError("TEP ground truth contains missing or duplicate sample IDs")
    return frame


def load_tep_five_class_alarm_runs(
    path: str | Path,
    *,
    disturbances: tuple[str, ...] | list[str] | None = None,
) -> tuple[TEPAlarmRun, ...]:
    """Load the official 1,000-sample TEP five-class alarm ZIP without extraction."""

    source = Path(path)
    allowed = set(disturbances) if disturbances is not None else None
    records: list[TEPAlarmRun] = []
    expected_names: tuple[str, ...] | None = None
    with zipfile.ZipFile(source) as archive:
        truth = _ground_truth(archive)
        truth_by_number: dict[int, dict[str, object]] = {}
        positions: dict[int, int] = {}
        for row in truth.to_dict(orient="records"):
            sample_match = re.fullmatch(r"sample_(\d+)", str(row["sample ID"]))
            if sample_match is None:
                raise ValueError(f"invalid TEP sample ID: {row['sample ID']}")
            sample_number = int(sample_match.group(1))
            class_label = int(row["class label"])
            if sample_number in truth_by_number:
                raise ValueError(f"duplicate numeric TEP sample ID: {sample_number}")
            positions[class_label] = positions.get(class_label, 0) + 1
            truth_by_number[sample_number] = {
                **row,
                "class position": positions[class_label],
            }

        observed_samples: set[int] = set()
        for member in sorted(archive.namelist()):
            match = _CSV_MEMBER.fullmatch(member.replace("\\", "/"))
            if match is None:
                continue
            class_label = int(match.group("class_label"))
            sample_number = int(match.group("sample"))
            metadata = truth_by_number.get(sample_number)
            if metadata is None or int(metadata["class label"]) != class_label:
                raise ValueError(f"TEP CSV/ground-truth mismatch: {member}")
            disturbance = str(metadata["disturbance name"])
            if sample_number in observed_samples:
                raise ValueError(f"duplicate TEP alarm CSV for sample {sample_number}")
            observed_samples.add(sample_number)
            if allowed is not None and disturbance not in allowed:
                continue
            with archive.open(member) as stream:
                frame = pd.read_csv(stream)
            if tuple(frame.columns[:2]) != ("Timestamp", "Minutes"):
                raise ValueError(f"TEP alarm CSV must start with time columns: {member}")
            alarm_frame = frame.iloc[:, 2:]
            names = tuple(map(str, alarm_frame.columns))
            if not names or alarm_frame.columns.duplicated().any():
                raise ValueError(f"TEP alarm columns are empty or duplicated: {member}")
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ValueError(f"TEP alarm columns differ: {member}")
            minutes = frame["Minutes"].to_numpy(dtype=float)
            if not np.array_equal(minutes, np.arange(len(frame), dtype=float)):
                raise ValueError(f"TEP minute grid is not consecutive: {member}")
            records.append(
                TEPAlarmRun(
                    disturbance=disturbance,
                    class_label=class_label,
                    sample_number=sample_number,
                    class_position=int(metadata["class position"]),
                    min_scaling=float(metadata["min. scaling"]),
                    max_scaling=float(metadata["max. scaling"]),
                    alarm_states=alarm_frame.to_numpy(),
                    alarm_names=names,
                )
            )
    if observed_samples != set(truth_by_number):
        missing = sorted(set(truth_by_number) - observed_samples)
        raise ValueError(f"TEP archive is missing ground-truth samples: {missing[:10]}")
    if not records:
        raise ValueError("TEP alarm selection produced no runs")
    return tuple(sorted(records, key=lambda run: (run.class_label, run.sample_number)))


def build_tep_five_class_split(
    runs: tuple[TEPAlarmRun, ...] | list[TEPAlarmRun],
    *,
    train_per_class: int,
    calibration_per_class: int,
    test_per_class: int,
    random_state: int,
    representation: str = "state",
) -> TEPAlarmSplit:
    """Create a seeded, stratified split whose atomic group is one simulation sample."""

    if not runs:
        raise ValueError("TEP split requires runs")
    counts = (int(train_per_class), int(calibration_per_class), int(test_per_class))
    if any(count <= 0 for count in counts):
        raise ValueError("TEP split counts must be positive")
    alarm_names = runs[0].alarm_names
    shape = runs[0].alarm_states.shape
    if any(run.alarm_names != alarm_names or run.alarm_states.shape != shape for run in runs):
        raise ValueError("TEP runs must share alarm columns and matrix dimensions")
    rng = np.random.default_rng(int(random_state))
    partitions: dict[str, list[TEPAlarmRun]] = {
        "train": [],
        "calibration": [],
        "test": [],
    }
    for label in sorted({run.disturbance for run in runs}):
        class_runs = sorted(
            (run for run in runs if run.disturbance == label),
            key=lambda run: run.sample_number,
        )
        if sum(counts) != len(class_runs):
            raise ValueError(
                f"TEP split counts must consume every sample in {label}: "
                f"requested {sum(counts)}, available {len(class_runs)}"
            )
        shuffled = [class_runs[index] for index in rng.permutation(len(class_runs))]
        train_stop = counts[0]
        calibration_stop = train_stop + counts[1]
        partitions["train"].extend(shuffled[:train_stop])
        partitions["calibration"].extend(shuffled[train_stop:calibration_stop])
        partitions["test"].extend(shuffled[calibration_stop:])

    def arrays(name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        selected = partitions[name]
        X = np.stack([run.representation(representation).T for run in selected])
        y = np.asarray([run.disturbance for run in selected])
        ids = tuple(run.run_id for run in selected)
        return X, y, ids

    X_train, y_train, train_ids = arrays("train")
    X_calibration, y_calibration, calibration_ids = arrays("calibration")
    X_test, y_test, test_ids = arrays("test")
    if set(train_ids) & set(calibration_ids) or set(train_ids) & set(test_ids) or set(calibration_ids) & set(test_ids):
        raise RuntimeError("TEP split unexpectedly overlaps simulation samples")
    return TEPAlarmSplit(
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
        representation=representation,
        random_state=int(random_state),
    )
