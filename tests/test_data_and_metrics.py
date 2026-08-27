from pathlib import Path
import zipfile

import pytest

import numpy as np

from iia_benchmark.data import (
    ProcessRun,
    audit_pronto_archive,
    load_piade_alarm_events,
    load_piade_alarm_intervals,
    load_piade_alarm_sequences,
    load_skab_csv,
    load_tep_ascii,
)
from iia_benchmark.evaluation import prediction_set_metrics, robustness_degradation


def test_process_run_normalizes_lists() -> None:
    run = ProcessRun("r", [0, 1], [1, 2], [False, True], ("x",))
    assert run.values.shape == (2, 1)
    assert run.timestamps.dtype.kind == "f"


def test_piade_adapter_filters_no_alarm(tmp_path: Path) -> None:
    path = tmp_path / "piade.csv"
    path.write_text(
        "equipment_ID,alarm,start\ns_1,A_000,1.0\ns_1,A_123,2.0\ns_2,A_999,3.0\n",
        encoding="utf-8",
    )
    events = load_piade_alarm_events(path, equipment_id="s_1")
    assert [(event.timestamp, event.tag) for event in events] == [(2.0, "A_123")]


def test_piade_interval_and_sequence_adapters(tmp_path: Path) -> None:
    path = tmp_path / "piade.csv"
    path.write_text(
        "equipment_ID,alarm,start,end\n"
        "s_1,A_001,1.0,2.0\n"
        "s_1,A_002,3.0,4.0\n"
        "s_1,A_003,101.0,102.0\n"
        "s_2,A_004,5.0,6.0\n",
        encoding="utf-8",
    )
    intervals = load_piade_alarm_intervals(path, equipment_id="s_1")
    assert [(event.timestamp, event.tag, event.state) for event in intervals[:2]] == [
        (1.0, "A_001", 1),
        (2.0, "A_001", 0),
    ]
    sequences = load_piade_alarm_sequences(path, window_seconds=100.0)
    assert [[event.tag for event in sequence] for sequence in sequences["s_1"]] == [
        ["A_001", "A_002"]
    ]


def test_skab_adapter_keeps_point_labels_without_adjustment(tmp_path: Path) -> None:
    path = tmp_path / "experiment.csv"
    path.write_text(
        "datetime;sensor_a;sensor_b;anomaly;changepoint\n"
        "2020-01-01 00:00:00;1.0;2.0;0;0\n"
        "2020-01-01 00:00:01;1.5;2.5;1;1\n",
        encoding="utf-8",
    )
    run = load_skab_csv(path)
    assert run.feature_names == ("sensor_a", "sensor_b")
    assert run.values.shape == (2, 2)
    assert run.abnormal.tolist() == [False, True]


def test_uncertainty_and_robustness_metrics() -> None:
    result = prediction_set_metrics(["a", "b"], [["a"], ["a", "b"]])
    assert result["empirical_coverage"] == 1.0
    assert result["mean_prediction_set_size"] == 1.5
    assert robustness_degradation(0.9, 0.7) == pytest.approx(0.2)


def test_tep_adapter_detects_transposed_storage(tmp_path: Path) -> None:
    path = tmp_path / "d01_te.dat"
    np.savetxt(path, np.arange(52 * 5, dtype=float).reshape(52, 5))
    run = load_tep_ascii(path, fault_start=2, root_cause="IDV_01")
    assert run.values.shape == (5, 52)
    assert run.abnormal.tolist() == [False, False, True, True, True]


def test_pronto_archive_audit_inventory_and_traversal_rejection(tmp_path: Path) -> None:
    safe_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("PRONTO/Test1/process.csv", "time,value\n0,1\n")
        archive.writestr("PRONTO/Test1/alarms.txt", "Date,Time,Node\n")
    audit = audit_pronto_archive(safe_path, verify_crc=True)
    assert audit["files"] == 2
    assert audit["suffix_counts"] == {".csv": 1, ".txt": 1}
    assert audit["safe_to_extract"]
    assert audit["crc_verified"]

    unsafe_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_path, "w") as archive:
        archive.writestr("../outside.csv", "forbidden")
    unsafe_audit = audit_pronto_archive(unsafe_path)
    assert not unsafe_audit["safe_to_extract"]
    assert unsafe_audit["unsafe_members"][0]["reason"] == "empty_or_parent_path_component"
