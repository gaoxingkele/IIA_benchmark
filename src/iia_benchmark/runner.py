from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iia_benchmark.config import load_experiment_config, load_json_reference
from iia_benchmark.data import (
    make_synthetic_alarm_run,
    make_synthetic_causal_alarm_series,
    make_synthetic_floods,
    make_synthetic_multivariate_run,
)
from iia_benchmark.evaluation import binary_alarm_metrics, sequence_accuracy
from iia_benchmark.models import (
    ConvexHullNOZAlarm,
    TransferEntropyRanker,
    design_alarm,
    smith_waterman_similarity,
)


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def _load_references(root: Path, experiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field: load_json_reference(_resolve(root, experiment[field]))
        for field in ("system", "dataset", "split", "model", "metrics")
    }


def _range(specification: dict[str, Any]) -> np.ndarray:
    return np.linspace(
        float(specification["start"]),
        float(specification["stop"]),
        int(specification["num"]),
    )


def _run_univariate(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    run = make_synthetic_alarm_run(**dataset.get("generator", {}))
    values = run.values[:, int(dataset.get("value_column", 0))]
    search = model["search"]
    result = design_alarm(
        values,
        run.abnormal,
        thresholds=_range(search["thresholds"]),
        delays=search["delays"],
        deadbands=search["deadbands"],
        direction=model.get("direction", "high"),
        targets=tuple(model.get("targets", [0.05, 0.05, 10.0])),
        weights=tuple(model.get("weights", [1.0, 1.0, 0.25])),
    )
    alarm = result.model.predict(values)
    return {
        "parameters": {
            "threshold": result.model.threshold,
            "delay": result.model.delay,
            "deadband": result.model.deadband,
            "direction": result.model.direction,
        },
        "design_loss": result.loss,
        "metrics": binary_alarm_metrics(run.abnormal, alarm),
        "samples": len(values),
        "warning": "Smoke experiment tunes and evaluates on one synthetic run; it is not a leaderboard result.",
    }


def _run_flood_similarity(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episodes = make_synthetic_floods(**references["dataset"].get("generator", {}))
    truth: list[str] = []
    prediction: list[str] = []
    for index, episode in enumerate(episodes):
        candidates = [candidate for j, candidate in enumerate(episodes) if j != index]
        nearest = max(
            candidates,
            key=lambda candidate: smith_waterman_similarity(episode.tags(), candidate.tags()),
        )
        truth.append(episode.label or "unknown")
        prediction.append(nearest.label or "unknown")
    return {
        "metrics": {"nearest_neighbor_accuracy": sequence_accuracy(truth, prediction)},
        "episodes": len(episodes),
        "warning": "Synthetic smoke corpus; use run-group and open-set splits for reported results.",
    }


def _run_multivariate_noz(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    run = make_synthetic_multivariate_run(**dataset.get("generator", {}))
    estimator = ConvexHullNOZAlarm(
        false_alarm_fraction=float(model.get("false_alarm_fraction", 0.01))
    ).fit(run.values[~run.abnormal])
    alarm = estimator.predict(run.values)
    return {
        "metrics": binary_alarm_metrics(run.abnormal, alarm),
        "hull_facets": int(len(estimator.equations_)),
        "samples": len(run.timestamps),
        "warning": "Synthetic smoke corpus; use run-grouped held-out trajectories for reporting.",
    }


def _run_root_cause(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = references["dataset"]
    model = references["model"]
    series = make_synthetic_causal_alarm_series(**dataset.get("generator", {}))
    ranking = TransferEntropyRanker(
        max_lag=int(model.get("max_lag", 10)),
        permutations=int(model.get("permutations", 99)),
        significance=float(model.get("significance", 0.01)),
        seed=int(model.get("seed", 0)),
    ).rank(series, target="TARGET")
    return {
        "ranking": [
            {"tag": tag, "transfer_entropy": score, "lag": lag, "threshold": threshold}
            for tag, score, lag, threshold in ranking
        ],
        "top1_correct": bool(ranking and ranking[0][0] == "ROOT"),
        "warning": "Synthetic binary series; real reports require occurrence counts and surrogate policy.",
    }


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    root = path.parents[2]
    experiment = load_experiment_config(path)
    references = _load_references(root, experiment)
    task = experiment["task"]
    if task == "univariate_alarm_design":
        result = _run_univariate(references)
    elif task == "alarm_flood_similarity":
        result = _run_flood_similarity(references)
    elif task == "multivariate_noz":
        result = _run_multivariate_noz(references)
    elif task == "root_cause_transfer_entropy":
        result = _run_root_cause(references)
    else:
        raise ValueError(f"Unsupported runnable task: {task}")
    payload = {
        "experiment_id": experiment["id"],
        "task": task,
        "result": result,
        "config": str(path.relative_to(root)).replace("\\", "/"),
    }
    run_dir = _resolve(root, experiment["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an IIA benchmark experiment")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
