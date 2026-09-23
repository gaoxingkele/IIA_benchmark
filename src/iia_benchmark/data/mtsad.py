from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data/public_datasets/mtsad")
DEFAULT_CONFIG_DIR = Path("configs/datasets")


@dataclass(frozen=True)
class MTSADSplit:
    """One benchmark dataset with its native split and point labels.

    Raw values are kept exactly as published; standardisation is a separate,
    explicit step so that the preprocessing can never silently leak test
    statistics into training.
    """

    dataset: str
    train: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    feature_names: tuple[str, ...]
    source_paths: dict[str, str]
    repairs: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.train.ndim != 2 or self.test.ndim != 2:
            raise ValueError("train and test must be two-dimensional")
        if self.train.shape[1] != self.test.shape[1]:
            raise ValueError("train and test must share the feature count")
        if len(self.test_labels) != len(self.test):
            raise ValueError("point labels must align with the test split")
        if self.feature_names and len(self.feature_names) != self.train.shape[1]:
            raise ValueError("feature_names must match the feature count")

    @property
    def n_features(self) -> int:
        return self.train.shape[1]

    @property
    def anomaly_rate(self) -> float:
        return float(self.test_labels.mean())


def dataset_config_path(dataset: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> Path:
    return Path(config_dir) / f"mtsad_{dataset.lower()}.json"


def load_dataset_config(
    dataset: str, config_dir: str | Path = DEFAULT_CONFIG_DIR
) -> dict:
    path = dataset_config_path(dataset, config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no MTSAD dataset config at {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_npy_pair(
    train_path: Path, test_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(train_path, allow_pickle=False).astype(np.float64, copy=False)
    test = np.load(test_path, allow_pickle=False).astype(np.float64, copy=False)
    labels = np.load(label_path, allow_pickle=False).astype(bool, copy=False)
    return train, test, labels


def _load_psm(
    train_path: Path, test_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    label_frame = pd.read_csv(label_path)
    # Column 0 is the minute timestamp; the remaining columns are the metrics.
    train = train_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    test = test_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    labels = label_frame.iloc[:, 1].to_numpy(dtype=np.float64).astype(bool)
    # The reference PSMSegLoader applies numpy.nan_to_num, so the published
    # payload's non-finite entries become 0.0.  Count them so the substitution
    # is auditable rather than invisible.
    repair = {
        "train_nonfinite": int((~np.isfinite(train)).sum()),
        "test_nonfinite": int((~np.isfinite(test)).sum()),
    }
    train = np.nan_to_num(train)
    test = np.nan_to_num(test)
    return train, test, labels, repair


def _load_swat(
    train_path: Path, test_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the benchmark SWaT split exactly as the reference SWATSegLoader does."""

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    labels = test_frame.iloc[:, -1].to_numpy()
    train = train_frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    test = test_frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    if labels.dtype == object:
        normalised = np.char.lower(
            np.char.strip(labels.astype(str))
        )
        labels = np.where(normalised == "normal", 0.0, 1.0)
    labels = labels.astype(np.float64)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("SWaT labels must be binary after normalisation")
    return train, test, labels.astype(bool)


def load_mtsad_split(
    dataset: str,
    *,
    root: str | Path = DEFAULT_ROOT,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> MTSADSplit:
    """Load one benchmark dataset without standardising it."""

    config = load_dataset_config(dataset, config_dir)
    loader = config["loader"]
    base = Path(root) / config["directory"]
    source_paths = {
        key: str((base / name))
        for key, name in (config.get("files") or {}).items()
    }
    if loader == "npy":
        train, test, labels = _load_npy_pair(
            base / config["files"]["train"],
            base / config["files"]["test"],
            base / config["files"]["label"],
        )
        repairs: dict[str, int] = {}
    elif loader == "psm_csv":
        train, test, labels, repairs = _load_psm(
            base / config["files"]["train"],
            base / config["files"]["test"],
            base / config["files"]["label"],
        )
    elif loader == "swat_csv":
        train, test, labels = _load_swat(
            base / config["files"]["train"], base / config["files"]["test"]
        )
        repairs = {}
    else:
        raise ValueError(f"unsupported MTSAD loader: {loader}")

    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(f"{dataset}: non-finite values in the published split")
    expected = config.get("expected_shapes") or {}
    if expected:
        observed = [list(train.shape), list(test.shape)]
        if observed != [expected["train"], expected["test"]]:
            raise ValueError(
                f"{dataset}: split shapes {observed} do not match the benchmark "
                f"contract {[expected['train'], expected['test']]}"
            )
    return MTSADSplit(
        dataset=config["id"],
        train=np.ascontiguousarray(train),
        test=np.ascontiguousarray(test),
        test_labels=np.ascontiguousarray(labels),
        feature_names=tuple(
            config.get("feature_names") or [f"f{index}" for index in range(train.shape[1])]
        ),
        source_paths=source_paths,
        repairs=repairs,
    )


def standardise(
    split: MTSADSplit, *, ddof: int = 0
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Z-score both splits with statistics estimated on the training split only."""

    mean = split.train.mean(axis=0)
    std = split.train.std(axis=0, ddof=ddof)
    safe = np.where(std == 0, 1.0, std)
    train = (split.train - mean) / safe
    test = (split.test - mean) / safe
    stats = {"mean": mean, "std": std, "degenerate_channels": std == 0}
    return train, test, stats


def window_starts(length: int, window: int, step: int) -> np.ndarray:
    if window < 1 or step < 1:
        raise ValueError("window and step must be positive")
    if length < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, length - window + 1, step, dtype=np.int64)


def stack_windows(values: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Materialise sliding windows with shape (n_windows, window, n_features)."""

    starts = window_starts(len(values), window, step)
    if not len(starts):
        return np.empty((0, window, values.shape[1]), dtype=values.dtype)
    index = starts[:, None] + np.arange(window)[None, :]
    return values[index]


def expand_window_scores(scores: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Scatter per-window scores back to the flattened reference layout.

    The Time-Series-Library anomaly-detection harness concatenates the labels
    of every sliding window, so a window score is replicated ``window`` times.
    Reproducing that layout is what makes its point-adjusted F1 comparable.
    """

    if scores.ndim != 1:
        raise ValueError("window scores must be one-dimensional")
    return np.repeat(scores, window).astype(np.float64, copy=False)


def last_point_scores(
    scores: np.ndarray, *, length: int, window: int, step: int
) -> np.ndarray:
    """Assign each window score to the window's final timestamp.

    Windows overlap, so several windows vote for interior timestamps; the mean
    vote is used and leading timestamps that no window can score stay NaN,
    which keeps the uncovered prefix visible instead of silently zero-filled.
    """

    starts = window_starts(length, window, step)
    if len(starts) != len(scores):
        raise ValueError("score count does not match the window grid")
    totals = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)
    np.add.at(totals, starts + window - 1, scores)
    np.add.at(counts, starts + window - 1, 1)
    energy = np.full(length, np.nan, dtype=np.float64)
    covered = counts > 0
    energy[covered] = totals[covered] / counts[covered]
    return energy


def point_scores_from_windows(
    scores: np.ndarray,
    *,
    length: int,
    window: int,
    step: int,
    aggregation: str = "window_flatten",
) -> np.ndarray:
    """Aggregate per-window anomaly scores into the layout a protocol expects."""

    starts = window_starts(length, window, step)
    if len(starts) != len(scores):
        raise ValueError("score count does not match the window grid")
    if aggregation == "window_flatten":
        return expand_window_scores(scores, window, step)
    if aggregation == "last_point":
        return last_point_scores(scores, length=length, window=window, step=step)
    raise ValueError(f"unsupported aggregation: {aggregation}")
