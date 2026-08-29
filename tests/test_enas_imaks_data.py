from __future__ import annotations

from pathlib import Path
import zipfile

from iia_benchmark.data import (
    load_enas_event_log,
    load_imaks_causal_edges,
    load_imaks_sensor_data,
)


def test_enas_adapter_preserves_exception_rows_and_error_impulses(tmp_path: Path) -> None:
    source = tmp_path / "enas.csv"
    source.write_text(
        "Timestamp,C1,LS1,ME,HE,UE,PV\n"
        "2020-01-01 00:00:00,0,1,0,0,0,1\n"
        "2020-01-01 00:00:01,1,1,1,0,0,1\n"
        "2020-01-01 00:00:02,1,0,0,0,1,2\n",
        encoding="utf-8",
    )
    run = load_enas_event_log(source)
    assert run.signal_names == ("C1", "LS1")
    assert run.signal("C1").tolist() == [0, 1, 1]
    assert run.error("ME").tolist() == [0, 1, 0]
    assert run.production_variant.tolist() == [1, 1, 2]


def test_imaks_adapter_aligns_long_table_and_resolves_kg_edges(tmp_path: Path) -> None:
    source = tmp_path / "imaks.zip"
    sensor_rows = [
        "timestamp,sensor_id,value,anomaly_label,alarm_flag",
        "2026-01-01T00:00:00,S2,2.0,NORMAL,NONE",
        "2026-01-01T00:00:00,S1,1.0,NORMAL,NONE",
        "2026-01-01T00:00:30,S2,2.5,CORRELATED,WARNING",
        "2026-01-01T00:00:30,S1,1.5,DRIFT,WARNING",
    ]
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sensors/timeseries_annotated.csv", "\n".join(sensor_rows))
        archive.writestr("kg_seed/nodes.csv", "nodeId,name\nN1,S1\nN2,S2\n")
        archive.writestr(
            "kg_seed/edges.csv",
            "fromId,toId,type,ruleRef\nN1,N2,correlates_with,RULE-1\n",
        )
    data = load_imaks_sensor_data(source)
    assert data.sensor_names == ("S1", "S2")
    assert data.values.tolist() == [[1.0, 2.0], [1.5, 2.5]]
    assert data.sample_seconds == 30.0
    assert data.anomaly_state("S1").tolist() == [0, 1]
    assert load_imaks_causal_edges(source)[0].rule_reference == "RULE-1"
