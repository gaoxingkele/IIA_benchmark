from __future__ import annotations

import json

import numpy as np
import pytest

from iia_benchmark.data.mtsad import (
    load_mtsad_split,
    stack_windows,
    standardise,
    window_starts,
)


def write_dataset(tmp_path):
    root = tmp_path / "mtsad"
    config_dir = tmp_path / "configs"
    (root / "toy").mkdir(parents=True)
    (config_dir).mkdir(parents=True)
    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 3))
    test = rng.normal(size=(30, 3))
    labels = np.zeros(30, dtype=np.float32)
    labels[10:15] = 1.0
    np.save(root / "toy" / "train.npy", train)
    np.save(root / "toy" / "test.npy", test)
    np.save(root / "toy" / "label.npy", labels)
    (config_dir / "mtsad_toy.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "mtsad_toy",
                "family": "mtsad",
                "loader": "npy",
                "directory": "toy",
                "files": {"train": "train.npy", "test": "test.npy", "label": "label.npy"},
                "expected_shapes": {"train": [40, 3], "test": [30, 3]},
                "features": 3,
                "window": 5,
                "window_step": 1,
                "anomaly_ratio": 10.0,
            }
        ),
        encoding="utf-8",
    )
    return root, config_dir, train, test, labels


def test_load_split_round_trip(tmp_path) -> None:
    root, config_dir, train, test, labels = write_dataset(tmp_path)
    split = load_mtsad_split("toy", root=root, config_dir=config_dir)
    np.testing.assert_allclose(split.train, train)
    np.testing.assert_allclose(split.test, test)
    np.testing.assert_array_equal(split.test_labels, labels.astype(bool))
    assert split.anomaly_rate == pytest.approx(5 / 30)
    assert split.n_features == 3


def test_shape_contract_is_enforced(tmp_path) -> None:
    root, config_dir, _, _, _ = write_dataset(tmp_path)
    config = json.loads((config_dir / "mtsad_toy.json").read_text(encoding="utf-8"))
    config["expected_shapes"]["test"] = [31, 3]
    (config_dir / "mtsad_toy.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="shape"):
        load_mtsad_split("toy", root=root, config_dir=config_dir)


def test_standardise_uses_training_statistics_only(tmp_path) -> None:
    root, config_dir, _, _, _ = write_dataset(tmp_path)
    split = load_mtsad_split("toy", root=root, config_dir=config_dir)
    train_z, test_z, stats = standardise(split)
    np.testing.assert_allclose(train_z.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(stats["mean"], split.train.mean(axis=0))
    # Re-deriving the test split from the recorded training statistics must
    # reproduce the returned values exactly.
    safe_std = np.where(stats["std"] == 0, 1.0, stats["std"])
    np.testing.assert_allclose(
        (split.test - stats["mean"]) / safe_std, test_z, atol=1e-12
    )
    # Test statistics must not leak into the scaler.
    assert not np.allclose(stats["mean"], split.test.mean(axis=0))


def test_window_helpers() -> None:
    values = np.arange(10, dtype=float)[:, None]
    assert window_starts(10, 3, 2).tolist() == [0, 2, 4, 6]
    windows = stack_windows(values, 3, 2)
    assert windows.shape == (4, 3, 1)
    np.testing.assert_array_equal(windows[0].reshape(-1), [0, 1, 2])
    assert stack_windows(values, 11, 1).shape == (0, 11, 1)
