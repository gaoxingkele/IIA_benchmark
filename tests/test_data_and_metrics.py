from pathlib import Path

import pytest

import numpy as np

from iia_benchmark.data import ProcessRun, load_piade_alarm_events, load_tep_ascii
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
