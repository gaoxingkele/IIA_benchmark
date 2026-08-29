#!/usr/bin/env python3
"""Run Book Chapter 4.3/4.4 numerical and controlled-recreation experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.models import (  # noqa: E402
    PLRContributionRCA,
    RecursiveBayesianAlarmRCA,
)


CONFIG = ROOT / "configs/experiments/book_chapter4_numeric_harness.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    sources = [
        ROOT / "src/iia_benchmark/models/root_cause_book.py",
        Path(__file__),
        CONFIG,
    ]
    return {
        "git_worktree_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources
        },
    }


def generate_plr_example(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Book equations (4.90)-(4.93), including unit-variance Gaussian noise."""

    rng = np.random.default_rng(seed)
    time_axis = np.arange(1, 3201, dtype=float)
    sine = np.sin(np.pi * time_axis / 100.0 - np.pi / 2.0)
    x1, x2 = np.empty(3200), np.empty(3200)
    x1[:1600] = 5.0 * sine[:1600] + 5.0 + rng.normal(size=1600)
    x2[:800] = 2.5 * sine[:800] + 2.5 + 1.5 * rng.normal(size=800)
    x2[800:1600] = 1.5 * rng.normal(size=800)
    x1[1600:2400] = (
        2.5 * sine[1600:2400] + 2.5 + 1.5 * rng.normal(size=800)
    )
    x1[2400:] = 1.5 * rng.normal(size=800)
    x2[1600:] = 5.0 * sine[1600:] + 5.0 + rng.normal(size=1600)
    y = np.zeros(3200)
    for index in range(3200):
        x1_delayed = x1[max(index - 10, 0)]
        x2_delayed = x2[max(index - 8, 0)]
        if index < 800:
            y[index] = 1.5 * x1_delayed + x2_delayed
        elif index < 1600:
            y[index] = 1.5 * x1_delayed
        elif index < 2400:
            y[index] = x1_delayed + 1.5 * x2_delayed
        else:
            y[index] = 1.5 * x2_delayed
    return x1, x2, y + rng.normal(size=3200)


def run_plr(seed: int) -> dict[str, object]:
    x1, x2, y = generate_plr_example(seed)
    boundaries = tuple(range(0, 3201, 100))
    contributions = PLRContributionRCA(
        max_segments=32, min_size=80, max_lag=15
    ).analyze(
        np.column_stack([x1, x2]), y, segment_boundaries=boundaries
    )
    rows = []
    correct = []
    for index, result in enumerate(contributions):
        expected = 0 if index < 16 else 1
        predicted = int(np.argmax(result.factors))
        if result.target_trend != 0:
            correct.append(predicted == expected)
        rows.append(
            {
                "segment": index + 1,
                "start": result.segment.start,
                "stop": result.segment.stop,
                "target_trend": result.target_trend,
                "source_trends": list(result.source_trends),
                "lags": list(result.lags),
                "factors": result.factors.tolist(),
                "expected_dominant_driver": "x1" if expected == 0 else "x2",
                "predicted_dominant_driver": "x1" if predicted == 0 else "x2",
            }
        )
    metrics = {
        "x1_delay": int(contributions[0].lags[0]),
        "x2_delay": int(contributions[0].lags[1]),
        "delay_absolute_error_sum": int(
            abs(contributions[0].lags[0] - 10) + abs(contributions[0].lags[1] - 8)
        ),
        "active_segments": int(sum(row["target_trend"] != 0 for row in rows)),
        "dominant_driver_accuracy": float(np.mean(correct)),
        "mean_x1_factor_segments_1_8": float(
            np.mean([row["factors"][0] for row in rows[:8]])
        ),
        "mean_x2_factor_segments_17_24": float(
            np.mean([row["factors"][1] for row in rows[16:24]])
        ),
    }
    return {
        "method": "book_4_4_plr_rca",
        "seed": seed,
        "protocol_fidelity": "P2_equation_defined_numerical_reproduction",
        "metrics": metrics,
        "activation": {
            "passed": metrics["delay_absolute_error_sum"] == 0
            and metrics["dominant_driver_accuracy"] >= 0.75,
            "beacon": "both published delays recovered and segment driver accuracy >= 0.75",
        },
        "segments": rows,
    }


def _set_interval(values: np.ndarray, start: int, stop: int, value: int) -> None:
    """Apply an inclusive one-based book interval to a zero-based array."""

    values[start - 1 : stop] = value


def run_recursive_bayesian() -> dict[str, object]:
    samples = 3200
    true_x1 = np.zeros(samples, dtype=int)
    true_x2 = np.zeros(samples, dtype=int)
    true_alarm = np.zeros(samples, dtype=int)
    _set_interval(true_x2, 662, 1001, 1)
    _set_interval(true_alarm, 662, 1001, 1)
    _set_interval(true_x1, 1250, 1900, 1)
    _set_interval(true_x2, 1350, 1600, 1)
    _set_interval(true_alarm, 1250, 1900, 1)
    _set_interval(true_alarm, 2530, 3000, 1)

    observed_x1, observed_x2, observed_alarm = (
        true_x1.copy(),
        true_x2.copy(),
        true_alarm.copy(),
    )
    for start, stop in ((2303, 2316), (2348, 2351)):
        _set_interval(observed_x1, start, stop, 1)
    for start, stop in (
        (1354, 1355),
        (1362, 1364),
        (1522, 1525),
        (1698, 1698),
        (1844, 1849),
        (2429, 2436),
        (2453, 2457),
    ):
        _set_interval(observed_x1, start, stop, 0)
    _set_interval(observed_x2, 1854, 1856, 1)
    for start, stop in ((2305, 2317), (2328, 2330), (2349, 2355), (2363, 2364)):
        _set_interval(observed_alarm, start, stop, 1)
    for start, stop in ((1367, 1368), (1524, 1527), (2425, 2432), (2436, 2439)):
        _set_interval(observed_alarm, start, stop, 0)

    def label(x1: int, x2: int, alarm: int) -> tuple[str, ...]:
        if not alarm:
            return ()
        active = tuple(name for name, value in (("X1", x1), ("X2", x2)) if value)
        return active or ("unknown",)

    truth = tuple(
        label(x1, x2, alarm)
        for x1, x2, alarm in zip(true_x1, true_x2, true_alarm, strict=True)
    )
    naive = tuple(
        label(x1, x2, alarm)
        for x1, x2, alarm in zip(
            observed_x1, observed_x2, observed_alarm, strict=True
        )
    )
    model = RecursiveBayesianAlarmRCA(
        ["X1", "X2"], response_time_samples=20, initial_probability=0.0
    )
    decisions = model.infer_sequence(
        np.column_stack([observed_x1, observed_x2]), observed_alarm
    )
    nuisance_mask = (
        (observed_x1 != true_x1)
        | (observed_x2 != true_x2)
        | (observed_alarm != true_alarm)
    )
    stable_regions = {
        "x2_only": (700, 1000, ("X2",)),
        "coexisting": (1400, 1580, ("X1", "X2")),
        "x1_only": (1700, 1840, ("X1",)),
        "unknown": (2580, 2980, ("unknown",)),
    }
    region_accuracy = {
        name: float(np.mean([value == expected for value in decisions[start:stop]]))
        for name, (start, stop, expected) in stable_regions.items()
    }
    metrics = {
        "overall_online_accuracy": float(
            np.mean([prediction == expected for prediction, expected in zip(decisions, truth, strict=True)])
        ),
        "naive_overall_accuracy": float(
            np.mean([prediction == expected for prediction, expected in zip(naive, truth, strict=True)])
        ),
        "nuisance_sample_accuracy": float(
            np.mean(
                [
                    decisions[index] == truth[index]
                    for index in np.flatnonzero(nuisance_mask)
                ]
            )
        ),
        "naive_nuisance_sample_accuracy": float(
            np.mean(
                [naive[index] == truth[index] for index in np.flatnonzero(nuisance_mask)]
            )
        ),
        "nuisance_samples": int(np.sum(nuisance_mask)),
        "region_accuracy": region_accuracy,
        "x2_detection_delay_samples": int(
            next(index - 661 for index in range(661, 1001) if decisions[index] == ("X2",))
        ),
        "unknown_detection_delay_samples": int(
            next(
                index - 2529
                for index in range(2529, 3000)
                if decisions[index] == ("unknown",)
            )
        ),
    }
    return {
        "method": "book_4_3_recursive_bn",
        "seed": 0,
        "protocol_fidelity": "P1_controlled_recreation_source_time_series_unavailable",
        "metrics": metrics,
        "activation": {
            "passed": min(region_accuracy.values()) >= 0.95
            and metrics["nuisance_sample_accuracy"]
            > metrics["naive_nuisance_sample_accuracy"],
            "beacon": "all stable root states >=0.95 accuracy and nuisance accuracy exceeds naive state lookup",
        },
        "evidence_boundary": (
            "The event logic and listed nuisance intervals are reproduced from the book, "
            "but the source thermal-plant signals are unavailable; exact paper-score credit is denied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    run_name = out_dir.name
    plan = json.loads(CONFIG.read_text(encoding="utf-8"))["runs"]
    if run_name not in plan:
        raise ValueError(f"out_dir basename must be one of {sorted(plan)}")
    started = datetime.now(timezone.utc)
    execution_provenance = provenance()
    before = time.perf_counter()
    run = plan[run_name]
    result = run_plr(int(run["seed"])) if run["method"] == "plr" else run_recursive_bayesian()
    duration = time.perf_counter() - before
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG),
        "git_revision": revision,
        "execution_provenance": execution_provenance,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": duration,
        "result": result,
    }
    final_path = out_dir / "final_info.json"
    final_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_name": run_name, "metrics": result["metrics"], "activation": result["activation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
