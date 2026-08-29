#!/usr/bin/env python3
"""Execute Book Chapter 4.2 IGTE/IGDTE on three acquired datasets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iia_benchmark.data import (  # noqa: E402
    load_pronto_merged_csv,
    load_skab_csv,
    load_tep_ascii,
)
from iia_benchmark.models import (  # noqa: E402
    cluster_information_granules,
    clustered_surrogate_threshold,
    discrete_direct_transfer_entropy,
    discrete_transfer_entropy,
    information_granules,
    lagged_correlation_delay,
)


CONFIG = ROOT / "configs/experiments/book_chapter4_igte_multidataset.json"

TEP_REFERENCE_IGTE = {
    ("V1", "V4"),
    ("V1", "V8"),
    ("V4", "V16"),
    ("V4", "V18"),
    ("V4", "V38"),
    ("V4", "V50"),
    ("V8", "V4"),
    ("V8", "V31"),
    ("V16", "V31"),
    ("V18", "V38"),
    ("V18", "V50"),
    ("V50", "V18"),
}
TEP_REFERENCE_PRUNED = {
    ("V1", "V4"),
    ("V4", "V38"),
    ("V4", "V50"),
    ("V16", "V31"),
}
TEP_REFERENCE_IGDTE = TEP_REFERENCE_IGTE - TEP_REFERENCE_PRUNED


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
        ROOT / "src/iia_benchmark/data/tep.py",
        ROOT / "src/iia_benchmark/data/pronto.py",
        ROOT / "src/iia_benchmark/data/skab.py",
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


def contiguous_label_slice(labels: np.ndarray, label: str, samples: int) -> slice:
    mask = labels == label
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    candidates = [(int(stop - start), int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]
    if not candidates:
        raise ValueError(f"label {label!r} is absent")
    _, start, stop = max(candidates)
    if stop - start < samples:
        raise ValueError(f"label {label!r} has fewer than {samples} contiguous samples")
    return slice(start, start + samples)


def distribution_prior(
    values: np.ndarray,
    names: tuple[str, ...],
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, object]:
    standard_deviations = np.std(values, axis=0)
    shifts = np.abs(np.mean(after, axis=0) - np.mean(before, axis=0)) / np.maximum(
        np.std(before, axis=0), 1e-12
    )
    return {
        "samples": int(len(values)),
        "features": len(names),
        "finite": bool(np.isfinite(values).all()),
        "minimum_standard_deviation": float(np.min(standard_deviations)),
        "constant_features": [
            name for name, value in zip(names, standard_deviations, strict=True) if value <= 1e-12
        ],
        "median_absolute_standardized_shift": float(np.median(shifts)),
        "maximum_absolute_standardized_shift": float(np.max(shifts)),
    }


def load_episodes(dataset: str, config: dict[str, object]) -> tuple[list[dict[str, object]], list[Path]]:
    specification = config["datasets"][dataset]
    episodes: list[dict[str, object]] = []
    paths: list[Path] = []
    if dataset == "tep_idv1":
        path = ROOT / specification["path"]
        run = load_tep_ascii(path, fault_start=160)
        start, stop = specification["slice"]
        indices = [run.feature_names.index(name) for name in specification["loader_features"]]
        values = run.values[start:stop, indices]
        names = tuple(specification["features"])
        fault_start = int(specification["fault_start_in_slice"])
        episodes.append(
            {
                "id": "TEP_IDV1_900",
                "group": "IDV1",
                "values": values,
                "names": names,
                "prior": distribution_prior(
                    values, names, values[:fault_start], values[fault_start:]
                ),
            }
        )
        paths.append(path)
    elif dataset == "pronto":
        discovered = {
            path.stem: path for path in sorted(ROOT.glob(specification["glob"]))
        }
        for run_id, label in specification["episodes"]:
            path = discovered[run_id]
            run = load_pronto_merged_csv(path)
            names = tuple(specification["features"])
            indices = [run.process_names.index(name) for name in names]
            selected = contiguous_label_slice(
                run.labels, label, int(specification["episode_samples"])
            )
            values = run.process_values[selected][:, indices]
            normal = run.process_values[run.labels == "Normal"][:, indices]
            episodes.append(
                {
                    "id": f"{run_id}_{label.replace(' ', '_')}",
                    "group": label,
                    "values": values,
                    "names": names,
                    "prior": distribution_prior(values, names, normal, values),
                }
            )
            paths.append(path)
    elif dataset == "skab":
        for relative in specification["paths"]:
            path = ROOT / relative
            run = load_skab_csv(path)
            anomaly = np.flatnonzero(run.abnormal)
            if not len(anomaly):
                raise ValueError(f"{relative} contains no anomaly")
            samples = int(specification["episode_samples"])
            pre = int(specification["pre_anomaly_samples"])
            start = max(int(anomaly[0]) - pre, 0)
            start = min(start, len(run.values) - samples)
            values = run.values[start : start + samples]
            local_fault = int(anomaly[0]) - start
            names = tuple(run.feature_names)
            group = Path(relative).parent.name
            episodes.append(
                {
                    "id": f"{group}_{Path(relative).stem}",
                    "group": group,
                    "values": values,
                    "names": names,
                    "prior": distribution_prior(
                        values, names, values[:local_fault], values[local_fault:]
                    ),
                }
            )
            paths.append(path)
    else:
        raise ValueError(f"unsupported dataset {dataset}")
    return episodes, paths


def infer_graph(
    values: np.ndarray,
    names: tuple[str, ...],
    model: dict[str, object],
    seed: int,
) -> dict[str, object]:
    order = int(model["order"])
    max_raw_lag = int(model["max_raw_lag"])
    min_window = int(model["min_window_size"])
    min_samples = int(model["min_samples"])
    simulations = int(model["surrogate_simulations"])
    significance = float(model["significance"])
    cache: dict[tuple[int, int], np.ndarray] = {}

    def labels(feature: int, window: int) -> np.ndarray:
        key = (feature, window)
        if key not in cache:
            cache[key] = cluster_information_granules(
                information_granules(values[:, feature], window),
                min_samples=min_samples,
                xi=float(model["optics_xi"]),
            )
        return cache[key]

    candidates = []
    pair_diagnostics = []
    edge_index = 0
    for source in range(len(names)):
        for target in range(len(names)):
            if source == target:
                continue
            raw_lag, raw_correlation, raw_threshold = lagged_correlation_delay(
                values[:, source], values[:, target], max_lag=max_raw_lag
            )
            window = max(min_window, int(np.rint(max(raw_lag, min_window) / order)))
            xlabels, ylabels = labels(source, window), labels(target, window)
            score = discrete_transfer_entropy(
                xlabels,
                ylabels,
                lag=int(model["granular_lag"]),
                source_horizon=order,
                target_horizon=order,
            )
            threshold = clustered_surrogate_threshold(
                xlabels,
                ylabels,
                lag=int(model["granular_lag"]),
                order=order,
                simulations=simulations,
                significance=significance,
                seed=seed + edge_index,
            )
            edge_index += 1
            pair_diagnostics.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "igte": float(score),
                    "threshold": float(threshold),
                    "margin": float(score - threshold),
                    "raw_lag": int(raw_lag),
                    "window_size": int(window),
                }
            )
            if score > threshold:
                candidates.append(
                    {
                        "source": names[source],
                        "target": names[target],
                        "source_index": source,
                        "target_index": target,
                        "igte": float(score),
                        "threshold": float(threshold),
                        "raw_lag": int(raw_lag),
                        "raw_correlation": float(raw_correlation),
                        "raw_correlation_threshold": float(raw_threshold),
                        "window_size": int(window),
                    }
                )
    candidate_pairs = {(edge["source"], edge["target"]) for edge in candidates}
    for edge in candidates:
        source, target = edge["source"], edge["target"]
        intermediates = []
        for node in names:
            if node in {source, target}:
                continue
            mediator = (source, node) in candidate_pairs and (node, target) in candidate_pairs
            confounder = (node, source) in candidate_pairs and (node, target) in candidate_pairs
            if mediator or confounder:
                intermediates.append(node)
        direct = True
        conditional_scores = []
        for node in intermediates:
            source_index, target_index, intermediate_index = (
                names.index(source), names.index(target), names.index(node)
            )
            window = int(edge["window_size"])
            score = discrete_direct_transfer_entropy(
                labels(source_index, window),
                labels(target_index, window),
                labels(intermediate_index, window),
                source_lag=int(model["granular_lag"]),
                intermediate_lag=int(model["granular_lag"]),
                source_horizon=order,
                target_horizon=order,
                intermediate_horizon=order,
            )
            conditional_scores.append({"intermediate": node, "igdte": float(score)})
            if score <= edge["threshold"]:
                direct = False
        edge["intermediates"] = intermediates
        edge["conditional_scores"] = conditional_scores
        edge["direct"] = direct
        edge.pop("source_index")
        edge.pop("target_index")
    direct_edges = [edge for edge in candidates if edge["direct"]]
    root_scores = {}
    for node in names:
        outgoing = sum(edge["igte"] for edge in direct_edges if edge["source"] == node)
        incoming = sum(edge["igte"] for edge in direct_edges if edge["target"] == node)
        root_scores[node] = float(outgoing - incoming)
    ranking = sorted(root_scores, key=lambda node: (-root_scores[node], node))
    return {
        "significant_edges": len(candidates),
        "direct_edges": len(direct_edges),
        "pruned_edges": len(candidates) - len(direct_edges),
        "root_ranking": ranking,
        "root_scores": root_scores,
        "edges": candidates,
        "top_pair_diagnostics": sorted(
            pair_diagnostics, key=lambda row: row["margin"], reverse=True
        )[:12],
    }


def edge_set(graph: dict[str, object], *, direct: bool) -> set[tuple[str, str]]:
    return {
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if not direct or edge["direct"]
    }


def jaccard(left: Iterable[tuple[str, str]], right: Iterable[tuple[str, str]]) -> float:
    a, b = set(left), set(right)
    return float(len(a & b) / len(a | b)) if a | b else 1.0


def reference_metrics(inferred: set[tuple[str, str]], reference: set[tuple[str, str]]) -> dict[str, float]:
    true_positive = len(inferred & reference)
    precision = true_positive / len(inferred) if inferred else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "jaccard": jaccard(inferred, reference),
    }


def analyze_dataset(dataset: str, seed: int, config: dict[str, object]) -> dict[str, object]:
    episodes, paths = load_episodes(dataset, config)
    results = []
    for episode_index, episode in enumerate(episodes):
        before = time.perf_counter()
        graph = infer_graph(
            episode["values"], episode["names"], config["model"], seed + 1000 * episode_index
        )
        graph["wall_clock_seconds"] = time.perf_counter() - before
        results.append(
            {
                "episode_id": episode["id"],
                "group": episode["group"],
                "prior": episode["prior"],
                "graph": graph,
            }
        )
    graph_sets = [edge_set(result["graph"], direct=True) for result in results]
    within, cross = [], []
    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            score = jaccard(graph_sets[left], graph_sets[right])
            if results[left]["group"] == results[right]["group"]:
                within.append(score)
            else:
                cross.append(score)
    significant = [result["graph"]["significant_edges"] for result in results]
    direct = [result["graph"]["direct_edges"] for result in results]
    pruned = [result["graph"]["pruned_edges"] for result in results]
    metrics: dict[str, object] = {
        "episodes": len(results),
        "graph_activation_rate": float(np.mean([value > 0 for value in significant])),
        "mean_significant_edges": float(np.mean(significant)),
        "mean_direct_edges": float(np.mean(direct)),
        "mean_pruned_edges": float(np.mean(pruned)),
        "within_group_direct_edge_jaccard": float(np.mean(within)) if within else None,
        "cross_group_direct_edge_jaccard": float(np.mean(cross)) if cross else None,
        "mean_graph_wall_clock_seconds": float(
            np.mean([result["graph"]["wall_clock_seconds"] for result in results])
        ),
    }
    if dataset == "tep_idv1":
        inferred_igte = edge_set(results[0]["graph"], direct=False)
        inferred_igdte = edge_set(results[0]["graph"], direct=True)
        metrics["igte_reference"] = reference_metrics(inferred_igte, TEP_REFERENCE_IGTE)
        metrics["igdte_reference"] = reference_metrics(inferred_igdte, TEP_REFERENCE_IGDTE)
        metrics["published_root_v1_rank"] = int(
            results[0]["graph"]["root_ranking"].index("V1") + 1
        )
    specification = config["datasets"][dataset]
    activation = {
        "igte_passed": all(value > 0 for value in significant),
        "igdte_passed": all(value > 0 for value in pruned),
        "beacon": "every episode has a surrogate-significant IGTE edge and an IGDTE-pruned triangle edge",
    }
    activation["passed"] = activation["igte_passed"] and activation["igdte_passed"]
    return {
        "dataset": dataset,
        "dataset_family": specification["family"],
        "dataset_match": specification["match"],
        "protocol_fidelity": specification["protocol"],
        "seed": seed,
        "input_files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(paths))
        ],
        "metrics": metrics,
        "activation": activation,
        "reporting_status": (
            "TEP same-simulator near-exact graph comparison; root rank and published-edge metrics admitted"
            if dataset == "tep_idv1"
            else "structural transfer only; feature-level root truth absent, so top-k/MRR and causal precision are prohibited"
        ),
        "episodes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    run_name = out_dir.name
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if run_name not in config["runs"]:
        raise ValueError(f"out_dir basename must be one of {sorted(config['runs'])}")
    plan = config["runs"][run_name]
    started = datetime.now(timezone.utc)
    execution_provenance = provenance()
    before = time.perf_counter()
    result = analyze_dataset(plan["dataset"], int(plan["seed"]), config)
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
    print(
        json.dumps(
            {
                "run_name": run_name,
                "dataset": result["dataset"],
                "metrics": result["metrics"],
                "activation": result["activation"],
                "wall_clock_seconds": duration,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
