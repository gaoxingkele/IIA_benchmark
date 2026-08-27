"""Book Chapter 6 alarm visual-analytics data and standalone HTML report."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from math import cos, log10, pi, sin
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from iia_benchmark.data.schema import AlarmEpisode, AlarmEvent
from iia_benchmark.models.flood import smith_waterman_similarity


@dataclass(frozen=True)
class AlarmVisualAnalyticsReport:
    """Serializable fact layer consumed by every Chapter 6 view."""

    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return self.payload


@dataclass(frozen=True, order=True)
class OperatorResponse:
    """Time-stamped operator action optionally linked to an alarm tag."""

    timestamp: float
    action: str
    related_tag: str | None = None


def _bubble_transform(value: float) -> float:
    return 1.0 + log10(value) if value > 1.0 else max(0.0, value)


def performance_bubble_coordinates(
    average_alarm_rate: float, peak_alarm_rate: float, unique_alarms: int
) -> dict[str, float | str]:
    """Chapter 6 Eqs. (6.1)--(6.2) and the five nested KPI zones."""

    if average_alarm_rate < 0 or peak_alarm_rate < 0 or unique_alarms < 0:
        raise ValueError("bubble metrics must be nonnegative")
    x = min(_bubble_transform(peak_alarm_rate), 5.0)
    y = min(_bubble_transform(average_alarm_rate), 4.0)
    nested_zones = (
        (2.0, 1.0, "predictive"),
        (3.0, 2.0, "robust"),
        (4.0, 3.0, "stable"),
        (5.0, 4.0, "reactive"),
    )
    zone = next((name for x_limit, y_limit, name in nested_zones if x <= x_limit and y <= y_limit), "overloaded")
    return {"x": x, "y": y, "area": float(unique_alarms), "zone": zone}


def _tag_summaries(events: Sequence[AlarmEvent], end_time: float) -> list[dict[str, object]]:
    by_tag: dict[str, list[AlarmEvent]] = {}
    for event in events:
        by_tag.setdefault(event.tag, []).append(event)
    summaries: list[dict[str, object]] = []
    for tag, tag_events in by_tag.items():
        active_since: float | None = None
        activation_count = 0
        occurrence_count = 0
        total_duration = 0.0
        for event in sorted(tag_events, key=lambda item: item.timestamp):
            if event.state == 1:
                occurrence_count += 1
                if active_since is None:
                    active_since = event.timestamp
                    activation_count += 1
            elif active_since is not None:
                total_duration += max(0.0, event.timestamp - active_since)
                active_since = None
        standing_duration = max(0.0, end_time - active_since) if active_since is not None else 0.0
        summaries.append(
            {
                "tag": tag,
                "alarm_count": occurrence_count,
                "independent_activations": activation_count,
                "closed_active_duration": total_duration,
                "standing_duration": standing_duration,
                "priority": min(event.priority for event in tag_events),
            }
        )
    return sorted(summaries, key=lambda item: (-int(item["alarm_count"]), str(item["tag"])))


def _active_intervals(events: Sequence[AlarmEvent], stop: float) -> list[dict[str, object]]:
    active: dict[str, tuple[float, int]] = {}
    intervals: list[dict[str, object]] = []
    for event in events:
        if event.state == 1 and event.tag not in active:
            active[event.tag] = (event.timestamp, event.priority)
        elif event.state == 0 and event.tag in active:
            onset, priority = active.pop(event.tag)
            intervals.append(
                {"tag": event.tag, "start": onset, "end": event.timestamp, "priority": priority}
            )
    intervals.extend(
        {"tag": tag, "start": onset, "end": stop, "priority": priority, "standing": True}
        for tag, (onset, priority) in active.items()
    )
    return sorted(intervals, key=lambda item: (float(item["start"]), str(item["tag"])))


def _burst_series(
    activations: Sequence[AlarmEvent],
    start: float,
    stop: float,
    window_seconds: float,
    sample_seconds: float,
    flood_start: int,
    flood_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    grid = np.arange(start, stop + sample_seconds, sample_seconds, dtype=float)
    if grid.size == 0:
        grid = np.array([start])
    timestamps = np.sort(np.asarray([event.timestamp for event in activations], dtype=float))
    right = np.searchsorted(timestamps, grid, side="right")
    left = np.searchsorted(timestamps, grid - window_seconds, side="left")
    counts = right - left
    states = np.zeros(grid.size, dtype=np.int8)
    active = False
    intervals: list[dict[str, float]] = []
    onset: float | None = None
    for index, count in enumerate(counts):
        if not active and count >= flood_start:
            active = True
            onset = float(grid[index])
        elif active and count < flood_end:
            active = False
            intervals.append({"start": float(onset), "end": float(grid[index])})
            onset = None
        states[index] = int(active)
    if active:
        intervals.append({"start": float(onset), "end": float(grid[-1])})
    return grid, counts.astype(float), states, intervals


def _fixed_bin_counts(
    activations: Sequence[AlarmEvent], tags: Sequence[str], start: float, stop: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(start, stop + width, width, dtype=float)
    if edges.size < 2:
        edges = np.array([start, start + width])
    elif edges[-1] <= stop:
        edges = np.append(edges, edges[-1] + width)
    matrix = np.zeros((len(tags), edges.size - 1), dtype=int)
    for index, tag in enumerate(tags):
        values = [event.timestamp for event in activations if event.tag == tag]
        matrix[index] = np.histogram(values, bins=edges)[0]
    return edges, matrix


def _jaccard_matrix(matrix: np.ndarray) -> np.ndarray:
    present = matrix > 0
    size = present.shape[0]
    result = np.eye(size, dtype=float)
    for row in range(size):
        for column in range(row + 1, size):
            union = int(np.sum(present[row] | present[column]))
            value = float(np.sum(present[row] & present[column]) / union) if union else 0.0
            result[row, column] = result[column, row] = value
    return result


def _similarity_matrix(
    episodes: Sequence[AlarmEpisode],
    similarity: Callable[[Sequence[str], Sequence[str]], float],
) -> np.ndarray:
    size = len(episodes)
    result = np.eye(size, dtype=float)
    sequences = [episode.tags() for episode in episodes]
    for row in range(size):
        for column in range(row + 1, size):
            value = float(similarity(sequences[row], sequences[column]))
            result[row, column] = result[column, row] = value
    return result


def build_alarm_visual_analytics(
    events: Sequence[AlarmEvent],
    *,
    episodes: Sequence[AlarmEpisode] = (),
    responses: Sequence[OperatorResponse] = (),
    console: str = "all",
    window_seconds: float = 600.0,
    sample_seconds: float = 60.0,
    flood_start: int = 10,
    flood_end: int = 5,
    top_n: int = 20,
    correlation_threshold: float = 0.5,
    similarity: Callable[[Sequence[str], Sequence[str]], float] = smith_waterman_similarity,
    spiral_beta: float = 1.0,
    spiral_scale: float = 1.0,
) -> AlarmVisualAnalyticsReport:
    """Compute all auditable fact tables for the Chapter 6 verification suite."""

    if min(window_seconds, sample_seconds, top_n) <= 0 or not 0 <= correlation_threshold <= 1:
        raise ValueError("window/sample/top_n must be positive and threshold must be in [0, 1]")
    if flood_start <= flood_end or flood_end < 0:
        raise ValueError("flood_start must be greater than flood_end >= 0")
    ordered = tuple(sorted(events, key=lambda item: (item.timestamp, item.tag, item.state)))
    if not ordered:
        raise ValueError("at least one alarm event is required")
    activations = tuple(event for event in ordered if event.state == 1)
    start, stop = ordered[0].timestamp, ordered[-1].timestamp
    stop_for_bins = max(stop, start + sample_seconds)
    summaries = _tag_summaries(ordered, stop)
    active_intervals = _active_intervals(ordered, stop)
    top_tags = tuple(str(item["tag"]) for item in summaries[:top_n])
    grid, burst, flood_state, flood_intervals = _burst_series(
        activations, start, stop_for_bins, window_seconds, sample_seconds, flood_start, flood_end
    )
    bin_edges, count_matrix = _fixed_bin_counts(
        activations, top_tags, start, stop_for_bins, window_seconds
    )
    correlation = _jaccard_matrix(count_matrix)
    edges = [
        {"source": top_tags[row], "target": top_tags[column], "strength": float(correlation[row, column])}
        for row in range(len(top_tags))
        for column in range(row + 1, len(top_tags))
        if correlation[row, column] >= correlation_threshold
    ]
    duration_bins = max((stop_for_bins - start) / window_seconds, 1.0)
    average_rate = len(activations) / duration_bins
    peak_rate = float(np.max(burst)) if burst.size else 0.0
    bubble = performance_bubble_coordinates(average_rate, peak_rate, len(summaries))
    episode_values = tuple(episodes)
    episode_similarity = _similarity_matrix(episode_values, similarity)
    theta = 2.0 * pi * (grid - start) / 86_400.0 + 2.0 * spiral_beta * pi
    radius = spiral_scale * theta
    spiral = [
        {
            "timestamp": float(timestamp),
            "x": float(radial * cos(angle + pi / 2.0)),
            "y": float(radial * sin(angle + pi / 2.0)),
            "burst_rate": float(rate),
            "flood": int(state),
        }
        for timestamp, radial, angle, rate, state in zip(grid, radius, theta, burst, flood_state)
    ]
    timeline = [
        {
            "event_index": index,
            "timestamp": event.timestamp,
            "offset_seconds": event.timestamp - start,
            "tag": event.tag,
            "state": event.state,
            "priority": event.priority,
        }
        for index, event in enumerate(ordered)
    ]
    midpoint = start + (stop - start) / 2.0
    previous_counts = {
        tag: sum(event.tag == tag and event.timestamp < midpoint for event in activations)
        for tag in top_tags
    }
    current_counts = {
        tag: sum(event.tag == tag and event.timestamp >= midpoint for event in activations)
        for tag in top_tags
    }
    previous_ranks = {
        tag: rank
        for rank, tag in enumerate(
            sorted(top_tags, key=lambda item: (-previous_counts[item], item)), start=1
        )
    }
    current_ranks = {
        tag: rank
        for rank, tag in enumerate(
            sorted(top_tags, key=lambda item: (-current_counts[item], item)), start=1
        )
    }
    bad_actor_tracking = [
        {
            "tag": tag,
            "previous_count": previous_counts[tag],
            "current_count": current_counts[tag],
            "previous_rank": previous_ranks[tag],
            "current_rank": current_ranks[tag],
            "rank_change": previous_ranks[tag] - current_ranks[tag],
        }
        for tag in sorted(top_tags, key=lambda item: current_ranks[item])
    ]
    analytics: list[dict[str, object]] = []
    for tag in top_tags:
        occurrences = np.asarray(
            [event.timestamp for event in activations if event.tag == tag], dtype=float
        )
        intervals_between = np.diff(occurrences).tolist() if occurrences.size > 1 else []
        durations = [
            float(interval["end"]) - float(interval["start"])
            for interval in active_intervals
            if interval["tag"] == tag
        ]
        analytics.append(
            {
                "tag": tag,
                "alarm_count_by_bin": count_matrix[top_tags.index(tag)].tolist(),
                "durations": durations,
                "inter_alarm_intervals": intervals_between,
            }
        )
    treemap_groups = []
    for priority in sorted({int(item["priority"]) for item in summaries}):
        children = [
            {"tag": item["tag"], "area": item["alarm_count"]}
            for item in summaries
            if int(item["priority"]) == priority
        ]
        treemap_groups.append(
            {"priority": priority, "area": sum(int(item["area"]) for item in children), "children": children}
        )
    response_values = tuple(sorted(responses))
    workflow_counts: dict[tuple[str, str], int] = {}
    response_flow: list[dict[str, object]] = []
    for response in response_values:
        candidate_tag = response.related_tag
        if candidate_tag is None:
            preceding = [event for event in activations if event.timestamp <= response.timestamp]
            candidate_tag = preceding[-1].tag if preceding else "UNLINKED"
        key = (candidate_tag, response.action)
        workflow_counts[key] = workflow_counts.get(key, 0) + 1
    for interval_index, interval in enumerate(active_intervals):
        related_responses = [
            {
                "action": response.action,
                "timestamp": response.timestamp,
                "offset_seconds": response.timestamp - float(interval["start"]),
            }
            for response in response_values
            if float(interval["start"]) <= response.timestamp <= float(interval["end"])
            and response.related_tag in {None, interval["tag"]}
        ]
        response_flow.append({"interval_index": interval_index, **interval, "responses": related_responses})
    flood_exposure = float(np.mean(flood_state)) if flood_state.size else 0.0
    top_contribution = (
        sum(int(item["alarm_count"]) for item in summaries[:10]) / max(1, len(activations))
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "chapter": "6",
        "source": {
            "book_doi": "10.1007/978-981-97-6516-4",
            "printed_pages": "381-417",
            "event_count": len(ordered),
            "activation_count": len(activations),
            "response_count": len(response_values),
            "episode_ids": [episode.episode_id for episode in episode_values],
        },
        "parameters": {
            "console": console,
            "window_seconds": window_seconds,
            "sample_seconds": sample_seconds,
            "flood_start": flood_start,
            "flood_end": flood_end,
            "top_n": top_n,
            "correlation_measure": "Jaccard on fixed-window activation incidence",
            "correlation_threshold": correlation_threshold,
            "similarity": getattr(similarity, "__name__", type(similarity).__name__),
            "spiral_beta": spiral_beta,
            "spiral_scale": spiral_scale,
        },
        "performance": {
            "average_alarm_rate_per_window": average_rate,
            "peak_alarm_rate_per_window": peak_rate,
            "unique_alarm_tags": len(summaries),
            "bubble": bubble,
            "bad_actors": summaries[:top_n],
            "hierarchical_treemap": treemap_groups,
            "alarm_analytics": analytics,
            "bad_actor_tracking": bad_actor_tracking,
            "layered_alarm_count_radar": {
                "tags": list(top_tags),
                "previous": [previous_counts[tag] for tag in top_tags],
                "current": [current_counts[tag] for tag in top_tags],
            },
            "performance_radar": {
                "metrics": ["average_rate", "peak_rate", "unique_tags", "flood_exposure", "top10_contribution"],
                "values": [average_rate, peak_rate, len(summaries), flood_exposure, top_contribution],
            },
        },
        "high_density_alarm_plot": {
            "tags": list(top_tags),
            "bin_edges": bin_edges.tolist(),
            "counts": count_matrix.tolist(),
        },
        "dynamic_3d_bar": {
            "time_axis": ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist(),
            "alarm_axis": list(top_tags),
            "count_axis": count_matrix.tolist(),
        },
        "burst_plot": {
            "timestamps": grid.tolist(),
            "counts": burst.tolist(),
            "flood_state": flood_state.tolist(),
            "flood_intervals": flood_intervals,
        },
        "related_alarms": {
            "tags": list(top_tags),
            "correlation_matrix": correlation.tolist(),
            "graph_edges": edges,
        },
        "alarm_response_workflow": {
            "nodes": sorted(
                {item for edge in workflow_counts for item in edge}
            ),
            "edges": [
                {"source": source, "target": target, "count": count}
                for (source, target), count in sorted(workflow_counts.items())
            ],
        },
        "alarm_response_event_flow": response_flow,
        "event_timeline": timeline,
        "alarm_flood_similarity": {
            "episode_ids": [episode.episode_id for episode in episode_values],
            "matrix": episode_similarity.tolist(),
        },
        "spiral": spiral,
    }
    return AlarmVisualAnalyticsReport(payload)


def _heatmap_table(labels: Sequence[str], values: Sequence[Sequence[float]], css_class: str) -> str:
    if not labels:
        return '<p class="empty">No episodes/tags available for this view.</p>'
    header = "".join(f"<th>{escape(str(label))}</th>" for label in labels)
    rows = []
    for label, row in zip(labels, values):
        cells = "".join(
            f'<td style="--v:{max(0.0, min(1.0, float(value))):.4f}">{float(value):.2f}</td>'
            for value in row
        )
        rows.append(f"<tr><th>{escape(str(label))}</th>{cells}</tr>")
    return f'<div class="table-scroll"><table class="heatmap {css_class}"><tr><th></th>{header}</tr>{"".join(rows)}</table></div>'


def _polyline(values: Sequence[float], width: int = 760, height: int = 170) -> str:
    if not values:
        return ""
    maximum = max(max(values), 1.0)
    points = " ".join(
        f"{index * width / max(1, len(values) - 1):.2f},{height - value * height / maximum:.2f}"
        for index, value in enumerate(values)
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="burst alarm rate"><polyline points="{points}" /></svg>'


def _radar_svg(
    labels: Sequence[str], first: Sequence[float], second: Sequence[float] | None = None
) -> str:
    labels = tuple(labels[:12])
    if len(labels) < 3:
        return '<p class="empty">At least three axes are required.</p>'
    series = [np.asarray(first[: len(labels)], dtype=float)]
    if second is not None:
        series.append(np.asarray(second[: len(labels)], dtype=float))
    scale = np.maximum.reduce([np.abs(item) for item in series])
    scale[scale == 0] = 1.0
    angles = np.arange(len(labels)) * 2.0 * pi / len(labels) - pi / 2.0
    axes = "".join(
        f'<line x1="180" y1="150" x2="{180 + 112*cos(angle):.2f}" y2="{150 + 112*sin(angle):.2f}"/><text x="{180 + 132*cos(angle):.2f}" y="{154 + 132*sin(angle):.2f}">{escape(str(label))}</text>'
        for label, angle in zip(labels, angles)
    )
    polygons = []
    for index, values in enumerate(series):
        radii = 105.0 * np.clip(values / scale, 0.0, 1.0)
        points = " ".join(
            f"{180 + radius*cos(angle):.2f},{150 + radius*sin(angle):.2f}"
            for radius, angle in zip(radii, angles)
        )
        polygons.append(f'<polygon class="r{index}" points="{points}"/>')
    return f'<svg class="radar" viewBox="0 0 360 300" role="img" aria-label="layered radar chart">{axes}{"".join(polygons)}</svg>'


def _render_html(report: AlarmVisualAnalyticsReport) -> str:
    payload = report.payload
    performance = payload["performance"]
    bubble = performance["bubble"]
    hdap = payload["high_density_alarm_plot"]
    related = payload["related_alarms"]
    similarity = payload["alarm_flood_similarity"]
    burst = payload["burst_plot"]
    bad_actor_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['tag']))}</td><td>{row['alarm_count']}</td>"
        f"<td>{row['independent_activations']}</td><td>{float(row['closed_active_duration']):.1f}</td>"
        f"<td>{float(row['standing_duration']):.1f}</td></tr>"
        for row in performance["bad_actors"]
    )
    matrix_max = max((max(row) for row in hdap["counts"]), default=1) or 1
    hdap_rows = "".join(
        f"<tr><th>{escape(tag)}</th>"
        + "".join(
            f'<td style="--v:{count / matrix_max:.4f}" title="{count} activations">{count}</td>'
            for count in row
        )
        + "</tr>"
        for tag, row in zip(hdap["tags"], hdap["counts"])
    )
    spiral_points = "".join(
        f'<circle cx="{380 + point["x"] * 8:.2f}" cy="{190 + point["y"] * 8:.2f}" r="2.5" class="s{point["flood"]}" />'
        for point in payload["spiral"]
    )
    treemap = "".join(
        f'<div class="tree-group"><b>P{group["priority"]}</b>'
        + "".join(
            f'<span style="--a:{max(1,int(child["area"]))}">{escape(str(child["tag"]))} · {child["area"]}</span>'
            for child in group["children"]
        )
        + "</div>"
        for group in performance["hierarchical_treemap"]
    )
    tracking_rows = "".join(
        f"<tr><td>{escape(str(row['tag']))}</td><td>{row['previous_count']}</td><td>{row['current_count']}</td><td>{row['rank_change']:+d}</td></tr>"
        for row in performance["bad_actor_tracking"]
    )
    analytics_rows = "".join(
        f"<tr><td>{escape(str(row['tag']))}</td><td>{len(row['durations'])}</td><td>{np.mean(row['durations']) if row['durations'] else 0:.2f}</td><td>{np.mean(row['inter_alarm_intervals']) if row['inter_alarm_intervals'] else 0:.2f}</td></tr>"
        for row in performance["alarm_analytics"]
    )
    timeline_rows = "".join(
        f"<tr><td>{row['event_index']}</td><td>{float(row['timestamp']):.2f}</td><td>{escape(str(row['tag']))}</td><td>{row['state']}</td><td>{row['priority']}</td></tr>"
        for row in payload["event_timeline"][:200]
    )
    workflow_rows = "".join(
        f"<tr><td>{escape(str(row['source']))}</td><td>{escape(str(row['target']))}</td><td>{row['count']}</td></tr>"
        for row in payload["alarm_response_workflow"]["edges"]
    ) or '<tr><td colspan="3" class="muted">No operator response records supplied.</td></tr>'
    graph_tags = related["tags"]
    graph_positions = {
        tag: (
            190 + 145 * cos(2 * pi * index / max(1, len(graph_tags))),
            170 + 125 * sin(2 * pi * index / max(1, len(graph_tags))),
        )
        for index, tag in enumerate(graph_tags)
    }
    graph_edges = "".join(
        f'<line x1="{graph_positions[edge["source"]][0]:.1f}" y1="{graph_positions[edge["source"]][1]:.1f}" x2="{graph_positions[edge["target"]][0]:.1f}" y2="{graph_positions[edge["target"]][1]:.1f}" style="--s:{edge["strength"]:.3f}" />'
        for edge in related["graph_edges"]
    )
    graph_nodes = "".join(
        f'<g><circle cx="{position[0]:.1f}" cy="{position[1]:.1f}" r="14"/><text x="{position[0]:.1f}" y="{position[1]+25:.1f}">{escape(tag)}</text></g>'
        for tag, position in graph_positions.items()
    )
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IIA Chapter 6 Verification Report</title><style>
:root{{--ink:#172033;--muted:#64748b;--panel:#fff;--line:#dbe4ee;--accent:#0f766e;--warm:#f59e0b}}
*{{box-sizing:border-box}} body{{margin:0;background:#f3f6f8;color:var(--ink);font:14px/1.45 Inter,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}} header{{padding:24px 0}} h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:17px;margin:0 0 16px}} .muted,.empty{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}} .panel{{grid-column:span 6;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;overflow:hidden}} .wide{{grid-column:span 12}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .card{{background:#eef7f5;border-radius:9px;padding:12px}} .card b{{display:block;font-size:22px}} .zone{{text-transform:uppercase;color:var(--accent);letter-spacing:.08em}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} .table-scroll{{overflow:auto}}
.heatmap td{{background:color-mix(in srgb,#dcfce7 calc((1 - var(--v))*100%),#dc2626 calc(var(--v)*100%));font-variant-numeric:tabular-nums}} .similarity td{{background:color-mix(in srgb,#eff6ff calc((1 - var(--v))*100%),#1d4ed8 calc(var(--v)*100%));color:#111}}
.hdap td{{padding:8px;background:color-mix(in srgb,#dcfce7 calc((1 - var(--v))*100%),#dc2626 calc(var(--v)*100%));text-align:center}} svg{{width:100%;height:auto;background:#f8fafc;border-radius:8px}} polyline{{fill:none;stroke:var(--accent);stroke-width:3}} .spiral .s0{{fill:#84cc16}} .spiral .s1{{fill:#dc2626}}
.treemap{{display:flex;gap:8px;align-items:stretch;min-height:170px}}.tree-group{{display:flex;flex:1;flex-wrap:wrap;align-content:flex-start;gap:5px;padding:10px;background:#eef2ff;border-radius:8px}}.tree-group b{{width:100%}}.tree-group span{{flex-grow:var(--a);background:#c7d2fe;padding:8px;border-radius:5px;font-size:11px}}.network line{{stroke:#0f766e;stroke-width:calc(1px + var(--s)*5px);opacity:.55}}.network circle{{fill:#fff;stroke:#0f766e;stroke-width:2}}.network text{{font-size:10px;text-anchor:middle;fill:#172033}}
.radar line{{stroke:#cbd5e1;stroke-width:1}}.radar text{{font-size:9px;text-anchor:middle}}.radar polygon{{stroke-width:2}}.radar .r0{{fill:#0f766e44;stroke:#0f766e}}.radar .r1{{fill:#f59e0b33;stroke:#f59e0b}}
.bubble{{position:relative;height:230px;border-left:1px solid var(--line);border-bottom:1px solid var(--line);background:linear-gradient(155deg,#dcfce7 0 23%,#fef9c3 23% 43%,#fde68a 43% 63%,#fed7aa 63% 82%,#fecaca 82%)}} .dot{{position:absolute;border-radius:50%;background:#0891b2aa;border:2px solid #0e7490;transform:translate(-50%,50%);display:grid;place-items:center;font-size:11px}}
@media(max-width:800px){{.panel,.wide{{grid-column:span 12}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<header><div class="muted">Intelligent Industrial Alarm Benchmark · Book Chapter 6</div><h1>Alarm Visual Analytics Verification</h1><div class="muted">Every view is generated from the embedded auditable fact layer; source event rows remain indexed in the JSON companion.</div></header>
<section class="grid"><article class="panel wide"><h2>Performance KPI views · Eqs. 6.1–6.2</h2><div class="cards"><div class="card"><span>Average / 10 min</span><b>{performance['average_alarm_rate_per_window']:.2f}</b></div><div class="card"><span>Peak / 10 min</span><b>{performance['peak_alarm_rate_per_window']:.0f}</b></div><div class="card"><span>Unique tags</span><b>{performance['unique_alarm_tags']}</b></div><div class="card"><span>Performance zone</span><b class="zone">{bubble['zone']}</b></div></div></article>
<article class="panel"><h2>Performance bubble</h2><div class="bubble"><span class="dot" style="left:{bubble['x']/5*100:.2f}%;bottom:{bubble['y']/4*100:.2f}%;width:{18+3*np.sqrt(bubble['area']):.1f}px;height:{18+3*np.sqrt(bubble['area']):.1f}px">{escape(str(payload['parameters']['console']))}</span></div></article>
<article class="panel"><h2>Top bad actors</h2><div class="table-scroll"><table><tr><th>Tag</th><th>Occurrences</th><th>Independent</th><th>Closed duration</th><th>Standing</th></tr>{bad_actor_rows}</table></div></article>
<article class="panel wide"><h2>Hierarchical alarm treemap · priority → tag</h2><div class="treemap">{treemap}</div></article>
<article class="panel"><h2>Alarm analytics graph</h2><table><tr><th>Tag</th><th>States</th><th>Mean duration</th><th>Mean interval</th></tr>{analytics_rows}</table></article>
<article class="panel"><h2>Two-period bad-actor tracking</h2><table><tr><th>Tag</th><th>Previous</th><th>Current</th><th>Rank Δ</th></tr>{tracking_rows}</table></article>
<article class="panel"><h2>Layered alarm-count radar</h2>{_radar_svg(performance['layered_alarm_count_radar']['tags'],performance['layered_alarm_count_radar']['previous'],performance['layered_alarm_count_radar']['current'])}</article>
<article class="panel"><h2>Multilayer performance radar</h2>{_radar_svg(performance['performance_radar']['metrics'],performance['performance_radar']['values'])}</article>
<article class="panel"><h2>Related-alarm graph</h2><svg class="network" viewBox="0 0 380 340">{graph_edges}{graph_nodes}</svg></article>
<article class="panel wide"><h2>High-density alarm plot / dynamic 3D-bar fact cube · 10-minute bins</h2><div class="table-scroll"><table class="hdap">{hdap_rows}</table></div></article>
<article class="panel wide"><h2>Alarm burst and flood state · Eqs. 6.15–6.16</h2>{_polyline(burst['counts'])}<p class="muted">Detected flood intervals: {len(burst['flood_intervals'])}; hysteresis thresholds: {payload['parameters']['flood_start']} / {payload['parameters']['flood_end']}.</p></article>
<article class="panel"><h2>Related-alarm Jaccard map · Eqs. 6.13–6.14</h2>{_heatmap_table(related['tags'],related['correlation_matrix'],'correlation')}</article>
<article class="panel"><h2>Alarm-flood similarity map · Eq. 6.17</h2>{_heatmap_table(similarity['episode_ids'],similarity['matrix'],'similarity')}</article>
<article class="panel"><h2>Alarm-response workflow</h2><table><tr><th>Alarm</th><th>Action</th><th>Count</th></tr>{workflow_rows}</table></article>
<article class="panel wide"><h2>Event timeline / alarm-response flow trace</h2><div class="table-scroll"><table><tr><th>Row</th><th>Timestamp</th><th>Tag</th><th>State</th><th>Priority</th></tr>{timeline_rows}</table></div></article>
<article class="panel wide"><h2>Alarm-flood spiral · Eqs. 6.18–6.20</h2><svg class="spiral" viewBox="0 0 760 380" role="img" aria-label="daily alarm flood spiral">{spiral_points}</svg></article>
<article class="panel wide"><h2>Traceability</h2><p>{payload['source']['event_count']} source rows, {payload['source']['activation_count']} activations, episode IDs: {escape(', '.join(payload['source']['episode_ids']) or 'none')}.</p></article></section>
<script id="iia-evidence" type="application/json">{embedded}</script></main></body></html>"""


def export_alarm_visual_report(
    report: AlarmVisualAnalyticsReport, output_directory: str | Path
) -> tuple[Path, Path]:
    """Export a self-contained HTML report and its exact JSON fact layer."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "alarm_visual_analytics.json"
    html_path = destination / "alarm_visual_analytics.html"
    json_path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_path.write_text(_render_html(report), encoding="utf-8")
    return html_path, json_path
