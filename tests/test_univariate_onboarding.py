import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iia_benchmark.adaptation import run_univariate_transfer
from iia_benchmark.data import load_univariate_transfer_config


def _write_config(tmp_path: Path, *, leaderboard: bool = True, duplicate_group: bool = False) -> Path:
    rng = np.random.default_rng(89)
    parts = {
        "normal_train": rng.normal(0.0, 1.0, 500),
        "normal_evaluation": rng.normal(0.0, 1.0, 400),
        "abnormal_calibration": rng.normal(4.0, 1.0, 240),
        "abnormal_evaluation": rng.normal(4.0, 1.0, 300),
    }
    specifications = {}
    for index, (name, values) in enumerate(parts.items()):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"signal": values}).to_csv(path, index=False)
        specifications[name] = {
            "loader": "csv",
            "path": path.name,
            "value_column": "signal",
            "filters": {},
            "row_start": 0,
            "row_stop": len(values),
            "group_id": "duplicate" if duplicate_group and index < 2 else name,
        }
    config = {
        "id": "test_transfer",
        "feature_name": "signal",
        "sample_period_seconds": 2.0,
        "leaderboard_eligible": leaderboard,
        "citation": {"title": "Test fixture", "url": "https://example.invalid"},
        "partitions": specifications,
        "adaptation": {
            "applicability_thresholds": {
                "normal_ks_adaptation": 0.2,
                "normal_median_shift_sd_adaptation": 0.5,
                "autocorrelation_block_calibration": 0.8,
                "minimum_block_auc": 0.6,
                "minimum_direction_consistency": 2.0 / 3.0,
                "chronological_blocks": 3,
            },
            "model": {
                "tail_probability": 0.05,
                "static_delay": 1,
                "reference_windows": [64, 128, 256],
                "delays": [1, 2, 3],
                "validation_fraction": 0.3,
                "block_size": 30,
                "target_point_false_alarm_rate": 0.05,
                "target_block_alarm_rate": 0.2,
                "block_weight": 0.25,
            },
            "uncertainty": {"block_size": 30, "draws": 50, "confidence": 0.9},
        },
        "output": "output.json",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_onboarding_loader_and_pipeline_are_configuration_driven(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    config, bundle = load_univariate_transfer_config(path, root=tmp_path)
    result = run_univariate_transfer(bundle, config["adaptation"], seed=97)
    assert result["status"] == "scored"
    assert result["calibration_applicability"]["scope"] == "training_and_calibration_only"
    assert result["held_out_posthoc_audit"]["scope"] == "held_out_posthoc_diagnostic_not_for_routing"
    assert result["empirical"]["f1"] > 0.8
    assert result["event_metrics"]["abnormal_event_recall"] == 1.0
    assert result["block_bootstrap"]["draws"] == 50
    normal_posterior = result["block_event_rate_posterior"][
        "normal_false_alarm_blocks"
    ]
    assert normal_posterior["blocks"] > 0


def test_onboarding_rejects_group_leakage_for_leaderboard_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path, duplicate_group=True)
    with pytest.raises(ValueError, match="distinct group_id"):
        load_univariate_transfer_config(path, root=tmp_path)


def test_onboarding_rejects_overlapping_rows_in_same_filtered_file(tmp_path: Path) -> None:
    path = _write_config(tmp_path, leaderboard=False)
    config = json.loads(path.read_text(encoding="utf-8"))
    shared = tmp_path / "shared.csv"
    pd.DataFrame({"signal": np.arange(20.0)}).to_csv(shared, index=False)
    for name, start, stop in (
        ("normal_train", 0, 12),
        ("normal_evaluation", 10, 20),
    ):
        config["partitions"][name].update(
            {"path": shared.name, "row_start": start, "row_stop": stop}
        )
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="partition overlap"):
        load_univariate_transfer_config(path, root=tmp_path)


def test_registered_skab_onboarding_report_is_real_but_not_leaderboard_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "experiments/reports/skab_univariate_onboarding_validation.json").read_text(
            encoding="utf-8"
        )
    )
    result = report["result"]
    assert result["status"] == "scored"
    assert result["selected_model"] == "block_calibrated_ecdf"
    assert not result["leaderboard_eligible"]
    assert result["empirical"]["f1"] > 0.4
    assert result["event_metrics"]["false_alarm_events_per_hour"] > 50.0
    abnormal_posterior = result["block_event_rate_posterior"][
        "abnormal_detection_blocks"
    ]
    assert abnormal_posterior["events"] > 0
    assert len(result["input_files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in result["input_files"])
