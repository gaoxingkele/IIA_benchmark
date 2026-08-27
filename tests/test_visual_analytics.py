import json

import numpy as np

from iia_benchmark.data import AlarmEpisode, AlarmEvent
from iia_benchmark.visualization import (
    OperatorResponse,
    build_alarm_visual_analytics,
    export_alarm_visual_report,
    performance_bubble_coordinates,
)


def _events() -> tuple[AlarmEvent, ...]:
    return (
        AlarmEvent(0.0, "A", 1, 1),
        AlarmEvent(1.0, "A", 1, 1),
        AlarmEvent(2.0, "B", 1, 2),
        AlarmEvent(3.0, "C", 1, 3),
        AlarmEvent(5.0, "A", 0, 1),
        AlarmEvent(7.0, "B", 0, 2),
        AlarmEvent(12.0, "A", 1, 1),
        AlarmEvent(15.0, "A", 0, 1),
        AlarmEvent(30.0, "B", 1, 2),
    )


def test_performance_bubble_equations_and_nested_zone() -> None:
    bubble = performance_bubble_coordinates(10.0, 100.0, 25)
    assert bubble["x"] == 3.0
    assert bubble["y"] == 2.0
    assert bubble["area"] == 25.0
    assert bubble["zone"] == "robust"


def test_visual_fact_layer_preserves_chatter_duration_and_event_trace() -> None:
    episodes = (
        AlarmEpisode("f1", tuple(event for event in _events() if event.state == 1), label="x"),
        AlarmEpisode("f2", (AlarmEvent(0, "A"), AlarmEvent(2, "B")), label="x"),
    )
    report = build_alarm_visual_analytics(
        _events(),
        episodes=episodes,
        responses=(OperatorResponse(4.0, "ACK", "A"),),
        window_seconds=10,
        sample_seconds=1,
        flood_start=3,
        flood_end=1,
        top_n=3,
    ).as_dict()
    actor_a = next(item for item in report["performance"]["bad_actors"] if item["tag"] == "A")
    assert actor_a["alarm_count"] == 3
    assert actor_a["independent_activations"] == 2
    assert actor_a["closed_active_duration"] == 8.0
    assert [item["event_index"] for item in report["event_timeline"]] == list(range(9))
    assert report["alarm_response_workflow"]["edges"] == [
        {"source": "A", "target": "ACK", "count": 1}
    ]
    assert report["performance"]["hierarchical_treemap"]
    assert report["performance"]["layered_alarm_count_radar"]["tags"]
    assert report["dynamic_3d_bar"]["alarm_axis"] == report["high_density_alarm_plot"]["tags"]


def test_burst_hysteresis_correlation_and_similarity_invariants() -> None:
    episodes = (
        AlarmEpisode("same-1", (AlarmEvent(0, "A"), AlarmEvent(1, "B"))),
        AlarmEpisode("same-2", (AlarmEvent(2, "A"), AlarmEvent(3, "B"))),
    )
    payload = build_alarm_visual_analytics(
        _events(),
        episodes=episodes,
        window_seconds=10,
        sample_seconds=1,
        flood_start=3,
        flood_end=1,
        top_n=3,
    ).as_dict()
    assert payload["burst_plot"]["flood_intervals"]
    correlation = np.asarray(payload["related_alarms"]["correlation_matrix"])
    similarity = np.asarray(payload["alarm_flood_similarity"]["matrix"])
    np.testing.assert_allclose(correlation, correlation.T)
    np.testing.assert_allclose(np.diag(correlation), 1.0)
    np.testing.assert_allclose(similarity, np.ones((2, 2)))


def test_export_creates_self_contained_html_and_json(tmp_path) -> None:
    report = build_alarm_visual_analytics(
        _events(), window_seconds=10, sample_seconds=1, flood_start=3, flood_end=1
    )
    html_path, json_path = export_alarm_visual_report(report, tmp_path)
    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Alarm Visual Analytics Verification" in html
    assert 'id="iia-evidence"' in html
    assert "High-density alarm plot" in html
    assert payload["source"]["event_count"] == len(_events())
    embedded = html.split('<script id="iia-evidence" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["schema_version"] == 1
