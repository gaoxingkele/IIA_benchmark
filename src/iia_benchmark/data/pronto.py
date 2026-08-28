from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProntoMergedRun:
    """One official aligned PRONTO alarm/process test-day table."""

    run_id: str
    alarm_states: np.ndarray
    process_values: np.ndarray
    labels: np.ndarray
    alarm_names: tuple[str, ...]
    process_names: tuple[str, ...]

    def __post_init__(self) -> None:
        alarms = np.asarray(self.alarm_states, dtype=np.int8)
        process = np.asarray(self.process_values, dtype=float)
        labels = np.asarray(self.labels, dtype=str)
        object.__setattr__(self, "alarm_states", alarms)
        object.__setattr__(self, "process_values", process)
        object.__setattr__(self, "labels", labels)
        if alarms.ndim != 2 or process.ndim != 2 or labels.ndim != 1:
            raise ValueError("PRONTO alarm/process/label arrays have invalid dimensions")
        if not (len(alarms) == len(process) == len(labels)) or not len(labels):
            raise ValueError("PRONTO arrays must have equal nonzero sample counts")
        if not np.isin(alarms, [0, 1]).all():
            raise ValueError("PRONTO alarm states must be binary")
        if not np.isfinite(process).all():
            raise ValueError("PRONTO process values must be finite")
        if len(self.alarm_names) != alarms.shape[1]:
            raise ValueError("alarm_names must match alarm columns")
        if len(self.process_names) != process.shape[1]:
            raise ValueError("process_names must match process columns")
        if np.any(np.char.str_len(labels) == 0):
            raise ValueError("PRONTO fault labels must be nonempty")


@dataclass(frozen=True)
class ProntoFaultWindowGroup:
    run_id: str
    segment_index: int
    label: str
    source_start: int
    source_stop: int
    windows: int
    train_windows: int
    purged_windows: int
    test_windows: int


@dataclass(frozen=True)
class ProntoFaultWindowSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    alarm_names: tuple[str, ...]
    groups: tuple[ProntoFaultWindowGroup, ...]


def load_pronto_merged_csv(
    path: str | Path, *, alarm_column_count: int = 12, label_column: str = "Fault"
) -> ProntoMergedRun:
    """Load and strictly validate an official aligned/labelled PRONTO CSV."""

    source = Path(path)
    frame = pd.read_csv(source)
    if alarm_column_count < 1 or label_column not in frame.columns:
        raise ValueError("invalid alarm_column_count or missing PRONTO label column")
    if frame.columns.duplicated().any() or len(frame.columns) <= alarm_column_count + 1:
        raise ValueError("PRONTO table has duplicate or insufficient columns")
    alarm_names = tuple(str(name) for name in frame.columns[:alarm_column_count])
    process_names = tuple(
        str(name) for name in frame.columns[alarm_column_count:] if name != label_column
    )
    return ProntoMergedRun(
        run_id=source.stem,
        alarm_states=frame.loc[:, alarm_names].to_numpy(),
        process_values=frame.loc[:, process_names].to_numpy(),
        labels=frame[label_column].astype(str).to_numpy(),
        alarm_names=alarm_names,
        process_names=process_names,
    )


def build_pronto_fault_window_split(
    runs: tuple[ProntoMergedRun, ...] | list[ProntoMergedRun],
    *,
    window_size: int,
    train_fraction: float,
    purge_windows: int = 1,
    excluded_labels: tuple[str, ...] = ("Normal",),
    alarm_representation: str = "state",
) -> ProntoFaultWindowSplit:
    """Create non-overlapping, chronologically purged windows per fault segment."""

    if not runs or window_size < 1 or not 0 < train_fraction < 1 or purge_windows < 0:
        raise ValueError("runs/window/split parameters are invalid")
    if alarm_representation not in {"state", "rising_edge"}:
        raise ValueError("alarm_representation must be 'state' or 'rising_edge'")
    alarm_names = tuple(sorted({name for run in runs for name in run.alarm_names}))
    train, test, y_train, y_test, groups = [], [], [], [], []
    excluded = set(excluded_labels)
    for run in runs:
        aligned = np.zeros((len(run.labels), len(alarm_names)), dtype=np.int8)
        target_columns = {name: index for index, name in enumerate(alarm_names)}
        for source_column, name in enumerate(run.alarm_names):
            aligned[:, target_columns[name]] = run.alarm_states[:, source_column]
        if alarm_representation == "rising_edge":
            previous = np.vstack(
                (np.zeros((1, aligned.shape[1]), dtype=np.int8), aligned[:-1])
            )
            aligned = np.maximum(aligned - previous, 0).astype(np.int8)
        starts = np.r_[0, np.flatnonzero(run.labels[1:] != run.labels[:-1]) + 1]
        stops = np.r_[starts[1:], len(run.labels)]
        for segment_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            label = str(run.labels[start])
            if label in excluded:
                continue
            window_count = int((stop - start) // window_size)
            minimum_windows = purge_windows + 2
            if window_count < minimum_windows:
                continue
            train_count = int(np.floor(window_count * train_fraction))
            train_count = min(max(1, train_count), window_count - purge_windows - 1)
            test_start = train_count + purge_windows
            for window_index in range(train_count):
                left = int(start + window_index * window_size)
                train.append(aligned[left : left + window_size].T)
                y_train.append(label)
            for window_index in range(test_start, window_count):
                left = int(start + window_index * window_size)
                test.append(aligned[left : left + window_size].T)
                y_test.append(label)
            groups.append(
                ProntoFaultWindowGroup(
                    run.run_id,
                    segment_index,
                    label,
                    int(start),
                    int(stop),
                    window_count,
                    train_count,
                    purge_windows,
                    window_count - test_start,
                )
            )
    if not train or not test:
        raise ValueError("PRONTO split produced no train or test windows")
    train_labels, test_labels = set(y_train), set(y_test)
    if len(train_labels) < 2 or train_labels != test_labels:
        raise ValueError("PRONTO train/test windows must cover the same multiple classes")
    return ProntoFaultWindowSplit(
        np.stack(train),
        np.asarray(y_train),
        np.stack(test),
        np.asarray(y_test),
        alarm_names,
        tuple(groups),
    )


def pronto_normal_train_evaluation_masks(
    labels: np.ndarray,
    *,
    normal_label: str = "Normal",
    train_fraction: float = 0.5,
    purge_samples: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every contiguous normal segment and retain all faults for evaluation."""

    values = np.asarray(labels, dtype=str)
    if values.ndim != 1 or values.size == 0 or not 0 < train_fraction < 1:
        raise ValueError("labels and train_fraction are invalid")
    if purge_samples < 0:
        raise ValueError("purge_samples must be nonnegative")
    train = np.zeros(values.size, dtype=bool)
    evaluate = values != normal_label
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    for start, stop in zip(starts, stops, strict=True):
        if values[start] != normal_label:
            continue
        length = int(stop - start)
        train_count = int(np.floor(length * train_fraction))
        test_start = int(start + train_count + purge_samples)
        if train_count < 1 or test_start >= stop:
            continue
        train[start : start + train_count] = True
        evaluate[test_start:stop] = True
    if not train.any() or not evaluate.any() or np.any(train & evaluate):
        raise ValueError("normal split must create disjoint nonempty train/evaluation masks")
    return train, evaluate


def _unsafe_archive_member(info: zipfile.ZipInfo) -> str | None:
    """Return the reason a ZIP member cannot be safely materialized."""

    normalized = info.filename.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(info.filename)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return "absolute_or_drive_path"
    if any(part in {"", ".."} for part in posix.parts):
        return "empty_or_parent_path_component"
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        return "symbolic_link"
    if info.flag_bits & 0x1:
        return "encrypted_member"
    return None


def audit_pronto_archive(path: str | Path, *, verify_crc: bool = False) -> dict:
    """Inventory a PRONTO ZIP before extraction.

    The inventory is intentionally independent of PRONTO's internal folder
    names.  It rejects path traversal, drive paths, symbolic links, and
    encrypted members before any extraction is allowed.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffixes: Counter[str] = Counter()
    unsafe: list[dict[str, str]] = []
    total_compressed = 0
    total_uncompressed = 0
    file_count = 0
    directory_count = 0
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        for info in infos:
            reason = _unsafe_archive_member(info)
            if reason:
                unsafe.append({"path": info.filename, "reason": reason})
            if info.is_dir():
                directory_count += 1
                continue
            file_count += 1
            suffixes[PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() or "<none>"] += 1
            total_compressed += info.compress_size
            total_uncompressed += info.file_size
        crc_failure = archive.testzip() if verify_crc and not unsafe else None
    return {
        "archive": source.as_posix(),
        "members": len(infos),
        "files": file_count,
        "directories": directory_count,
        "compressed_bytes": total_compressed,
        "uncompressed_bytes": total_uncompressed,
        "compression_ratio": (
            total_uncompressed / total_compressed if total_compressed else 0.0
        ),
        "suffix_counts": dict(sorted(suffixes.items())),
        "unsafe_members": unsafe,
        "safe_to_extract": not unsafe,
        "crc_verified": bool(verify_crc and not unsafe and crc_failure is None),
        "crc_failure": crc_failure,
    }


def extract_pronto_members(
    path: str | Path,
    destination: str | Path,
    *,
    prefixes: tuple[str, ...],
    maximum_total_bytes: int,
) -> tuple[Path, ...]:
    """Safely extract a bounded, prefix-selected PRONTO subset."""

    if not prefixes or maximum_total_bytes <= 0:
        raise ValueError("prefixes and maximum_total_bytes must be positive")
    normalized_prefixes = tuple(prefix.replace("\\", "/").rstrip("/") + "/" for prefix in prefixes)
    root = Path(destination).resolve()
    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        selected = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and any(info.filename.replace("\\", "/").startswith(prefix) for prefix in normalized_prefixes)
        ]
        if not selected:
            raise ValueError("no archive members matched the requested prefixes")
        unsafe = [
            (info.filename, reason)
            for info in selected
            if (reason := _unsafe_archive_member(info)) is not None
        ]
        if unsafe:
            raise ValueError(f"unsafe archive members selected: {unsafe}")
        total = sum(info.file_size for info in selected)
        if total > maximum_total_bytes:
            raise ValueError(
                f"selected members require {total} bytes, exceeding {maximum_total_bytes}"
            )
        extracted: list[Path] = []
        for info in selected:
            relative = PurePosixPath(info.filename.replace("\\", "/"))
            target = (root / Path(*relative.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"archive target escapes destination: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            extracted.append(target)
    return tuple(extracted)
