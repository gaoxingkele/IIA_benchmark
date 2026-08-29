from pathlib import Path
import zipfile
from io import BytesIO

import pytest

import numpy as np
from openpyxl import Workbook

from iia_benchmark.data import (
    ProcessRun,
    alarm_events_to_state_matrix,
    audit_pronto_archive,
    build_pronto_fault_window_split,
    build_fcc_alarm_split,
    build_tep_five_class_split,
    extract_pronto_members,
    load_piade_alarm_events,
    load_piade_alarm_intervals,
    load_piade_alarm_sequences,
    load_pronto_merged_csv,
    load_fcc_alarm_runs,
    load_fcc_timeseries_runs,
    load_tep_five_class_alarm_runs,
    pronto_normal_train_evaluation_masks,
    load_skab_csv,
    load_smd_alarm_events,
    load_tep_ascii,
)
from iia_benchmark.evaluation import multiclass_classification_metrics
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


def test_smd_alarm_event_adapter_filters_log_types(tmp_path: Path) -> None:
    path = tmp_path / "smd.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "WT01_logs.csv"
    sheet.append(
        [
            "Code",
            "Description",
            "Detected",
            "Device ack.",
            "Reset/Run",
            "Duration",
            "Event type",
            "Severity",
        ]
    )
    sheet.append([356, "alarm", "2020-01-01 01:02:03", None, None, None, "Alarm log (A)", 212])
    sheet.append([598, "warning", "2020-01-01 01:03:00", None, None, None, "Warning log (W)", 201])
    sheet.append([70, "system", "2020-01-01 01:04:00", None, None, None, "System log (S)", 101])
    workbook.save(path)
    events = load_smd_alarm_events(path)
    assert list(events) == ["WT01"]
    assert [(event.tag, event.priority) for event in events["WT01"]] == [
        ("356", 1),
        ("598", 2),
    ]
    alarms_only = load_smd_alarm_events(path, event_types=["Alarm log (A)"])
    assert [event.tag for event in alarms_only["WT01"]] == ["356"]
    with pytest.raises(ValueError, match="unknown turbines"):
        load_smd_alarm_events(path, turbines=["WT99"])


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


def test_alarm_events_replay_to_binary_state_matrix() -> None:
    from iia_benchmark.data import AlarmEvent

    events = [
        AlarmEvent(3.0, "B", 1),
        AlarmEvent(1.0, "A", 1),
        AlarmEvent(2.0, "A", 0),
        AlarmEvent(4.0, "B", 0),
    ]
    grid, names, states = alarm_events_to_state_matrix(
        events, sample_seconds=1.0, start=0.0, stop=4.0
    )
    assert grid.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert names == ("A", "B")
    assert states.tolist() == [
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0],
    ]


def test_pronto_subset_extraction_is_prefix_selected_and_bounded(tmp_path: Path) -> None:
    archive_path = tmp_path / "pronto.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("root/preprocessed/a.csv", "a,b\n1,2\n")
        archive.writestr("root/video/large.mp4", b"x" * 100)
    destination = tmp_path / "extracted"
    extracted = extract_pronto_members(
        archive_path,
        destination,
        prefixes=("root/preprocessed",),
        maximum_total_bytes=32,
    )
    assert [path.relative_to(destination).as_posix() for path in extracted] == [
        "root/preprocessed/a.csv"
    ]
    assert extracted[0].read_text(encoding="utf-8") == "a,b\n1,2\n"
    with pytest.raises(ValueError, match="exceeding"):
        extract_pronto_members(
            archive_path,
            destination,
            prefixes=("root/video",),
            maximum_total_bytes=10,
        )


def test_pronto_merged_adapter_and_purged_fault_window_split(tmp_path: Path) -> None:
    header = "A1,A2,P1,Fault\n"
    rows = []
    for label, first, second in (("F1", 1, 0), ("Normal", 0, 1), ("F2", 0, 1)):
        rows.extend(f"{first},{second},{index / 10},{label}\n" for index in range(12))
    source = tmp_path / "day.csv"
    source.write_text(header + "".join(rows), encoding="utf-8")
    run = load_pronto_merged_csv(source, alarm_column_count=2)
    assert run.alarm_states.shape == (36, 2)
    assert run.process_names == ("P1",)
    split = build_pronto_fault_window_split(
        [run], window_size=3, train_fraction=0.5, purge_windows=1
    )
    assert split.X_train.shape == (4, 2, 3)
    assert split.X_test.shape == (2, 2, 3)
    assert set(split.y_train) == set(split.y_test) == {"F1", "F2"}
    assert all(group.purged_windows == 1 for group in split.groups)
    activations = build_pronto_fault_window_split(
        [run],
        window_size=3,
        train_fraction=0.5,
        purge_windows=1,
        alarm_representation="rising_edge",
    )
    assert int(split.X_train.sum()) == 12
    assert int(activations.X_train.sum()) == 1
    # A2 is already standing during Normal, so F2 must not invent a new
    # activation at the fault-segment or window boundary.
    assert int(activations.X_train[activations.y_train == "F2"].sum()) == 0
    with pytest.raises(ValueError, match="alarm_representation"):
        build_pronto_fault_window_split(
            [run],
            window_size=3,
            train_fraction=0.5,
            alarm_representation="unknown",
        )


def test_multiclass_metrics_include_per_class_and_confusion_audit() -> None:
    result = multiclass_classification_metrics(
        ["a", "a", "b", "b"], ["a", "b", "b", "b"]
    )
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    assert result["confusion_matrix"]["a"]["b"] == 1
    assert result["per_class"]["b"]["recall"] == 1.0


def test_fcc_adapters_and_complete_run_split(tmp_path: Path) -> None:
    alarm_path = tmp_path / "alarm.zip"
    series_path = tmp_path / "series.zip"
    alarm_header = "A_low,A_high\n"
    with zipfile.ZipFile(alarm_path, "w") as archive:
        for label, offset in (("fault_a", 0), ("fault_b", 1)):
            for run_number in range(1, 7):
                rows = [
                    f"{int((sample + offset) % 2 == 0)},{int((sample + offset) % 2 == 1)}"
                    for sample in range(4)
                ]
                archive.writestr(
                    f"alarmseriesdata/{label}/{label}_run{run_number}_alarm.csv",
                    alarm_header + "\n".join(rows) + "\n",
                )
    with zipfile.ZipFile(series_path, "w") as archive:
        for label in ("fault_a", "fault_b"):
            for run_number in (1,):
                prefix = f"timeseriesdata/{label}/{label}_run{run_number}"
                archive.writestr(
                    f"{prefix}_process.csv", "Time,P1\n0,1.0\n1,2.0\n"
                )
                archive.writestr(
                    f"{prefix}_valves.csv", "Time,V1\n0,3.0\n1,4.0\n"
                )
                archive.writestr(
                    f"{prefix}_disturbances.csv",
                    "Time,Dist_V2\n0,0.5\n1,0.5\n",
                )

    runs = load_fcc_alarm_runs(alarm_path)
    assert len(runs) == 12
    assert runs[0].alarm_states.shape == (4, 2)
    assert runs[0].to_episode().label == "fault_a"
    rising = runs[0].representation("rising_edge")
    assert int(rising.sum()) == 4
    split = build_fcc_alarm_split(
        runs,
        train_run_numbers=(1, 2),
        calibration_run_numbers=(3, 4),
        test_run_numbers=(5, 6),
        representation="rising_edge",
    )
    assert split.X_train.shape == (4, 2, 4)
    assert set(split.y_train) == set(split.y_test) == {"fault_a", "fault_b"}
    with pytest.raises(ValueError, match="disjoint"):
        build_fcc_alarm_split(
            runs,
            train_run_numbers=(1, 2),
            calibration_run_numbers=(2, 3),
            test_run_numbers=(4, 5),
        )

    series = load_fcc_timeseries_runs(series_path)
    assert len(series) == 2
    assert series[0].process_names == ("P1",)
    assert series[0].timestamps.tolist() == [0.0, 1.0]
    assert series[0].root_disturbance == "Dist_fault"


def test_tep_five_class_adapter_and_seeded_group_split(tmp_path: Path) -> None:
    source = tmp_path / "tep_alarm.zip"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        ["sample ID", "class label", "disturbance name", "min. scaling", "max. scaling"]
    )
    for class_label, disturbance in ((0, "IDV1"), (1, "IDV2")):
        for class_position in range(1, 7):
            sample_number = class_label * 6 + class_position
            sheet.append(
                [f"sample_{sample_number}", class_label, disturbance, 0.6, 1.0]
            )
    workbook_bytes = BytesIO()
    workbook.save(workbook_bytes)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("root/ground_truth.xlsx", workbook_bytes.getvalue())
        for class_label in (0, 1):
            for class_position in range(1, 7):
                sample_number = class_label * 6 + class_position
                archive.writestr(
                    f"root/class_{class_label}/alarm_timeseries_{sample_number}.csv",
                    "Timestamp,Minutes,A_HI,A_LO\n"
                    + "\n".join(
                        f"{minute / 60:.4f},{minute},{int((minute + class_label) % 2 == 0)},{int((minute + class_label) % 2 == 1)}"
                        for minute in range(4)
                    )
                    + "\n",
                )
    runs = load_tep_five_class_alarm_runs(source)
    assert len(runs) == 12
    assert runs[0].disturbance == "IDV1"
    assert runs[-1].class_position == 6
    assert runs[0].representation("rising_edge").shape == (4, 2)
    assert runs[0].to_episode().root_cause == "IDV1"
    split = build_tep_five_class_split(
        runs,
        train_per_class=2,
        calibration_per_class=2,
        test_per_class=2,
        random_state=1103,
        representation="rising_edge",
    )
    assert split.X_train.shape == (4, 2, 4)
    assert set(split.y_test) == {"IDV1", "IDV2"}
    assert not set(split.train_run_ids) & set(split.test_run_ids)
    repeated = build_tep_five_class_split(
        runs,
        train_per_class=2,
        calibration_per_class=2,
        test_per_class=2,
        random_state=1103,
        representation="rising_edge",
    )
    assert repeated.train_run_ids == split.train_run_ids


def test_pronto_normal_split_is_purged_and_keeps_all_faults() -> None:
    labels = np.array(["Normal"] * 10 + ["Fault"] * 4 + ["Normal"] * 8)
    train, evaluate = pronto_normal_train_evaluation_masks(
        labels, train_fraction=0.5, purge_samples=2
    )
    np.testing.assert_array_equal(np.flatnonzero(train), [0, 1, 2, 3, 4, 14, 15, 16, 17])
    assert np.all(evaluate[10:14])
    assert not np.any(evaluate[[5, 6, 18, 19]])
    assert not np.any(train & evaluate)
