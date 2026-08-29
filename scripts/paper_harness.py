#!/usr/bin/env python3
"""Validate and summarize the IIA paper-harness planning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "paper_harness"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def build_audit() -> tuple[dict[str, Any], list[str]]:
    protocol = read_json(HARNESS / "protocol_freeze.v1.json")
    matrix = read_json(HARNESS / "experiment_matrix.v1.json")
    references = read_json(HARNESS / "reference_experiments.v1.json")
    book = read_json(ROOT / "configs/algorithms/book_algorithms.json")["algorithms"]
    sota = read_json(ROOT / "configs/algorithms/sota_algorithms.json")["algorithms"]
    sources = read_json(ROOT / "configs/datasets/public_sources.json")["sources"]
    tasks = read_json(ROOT / "configs/tasks/downstream_tasks.json")["tasks"]
    papers = read_json(ROOT / "papers/literature/registry.json")["papers"]

    issues: list[str] = []
    registered_algorithms = {item["id"] for item in [*book, *sota]}
    book_ids = {item["id"] for item in book}
    sota_ids = {item["id"] for item in sota}
    matrix_rows = matrix.get("algorithms", [])
    matrix_ids = [item.get("algorithm_id") for item in matrix_rows]
    matrix_id_set = set(matrix_ids)
    registered_datasets = {item["dataset_family"] for item in sources}
    registered_tasks = {item["id"] for item in tasks}
    registered_papers = {item["id"] for item in papers}
    reference_rows = references.get("papers", [])
    reference_ids = [item.get("paper_id") for item in reference_rows]
    reference_id_set = set(reference_ids)
    dataset_sets = matrix.get("dataset_sets", {})
    adapter_status = matrix.get("adapter_status", {})
    lanes = {item.get("id"): item for item in protocol.get("comparison_lanes", [])}

    if len(matrix_ids) != len(matrix_id_set):
        issues.append("duplicate algorithm_id in experiment matrix")
    missing_algorithms = registered_algorithms - matrix_id_set
    extra_algorithms = matrix_id_set - registered_algorithms
    if missing_algorithms:
        issues.append(f"registered algorithms missing from matrix: {sorted(missing_algorithms)}")
    if extra_algorithms:
        issues.append(f"unknown algorithms in matrix: {sorted(extra_algorithms)}")

    if set(adapter_status) != registered_datasets:
        issues.append(
            "adapter_status dataset families differ from public registry: "
            f"missing={sorted(registered_datasets - set(adapter_status))}; "
            f"extra={sorted(set(adapter_status) - registered_datasets)}"
        )

    pair_count = 0
    valid_pair_count = 0
    runnable_pair_count = 0
    runnable_valid_pair_count = 0
    pending_targets_by_family: dict[str, int] = {}
    algorithms_below_three_valid_datasets: list[str] = []
    used_papers: set[str] = set()
    for row in matrix_rows:
        algorithm_id = row.get("algorithm_id", "?")
        dataset_set_id = row.get("dataset_set")
        if dataset_set_id not in dataset_sets:
            issues.append(f"{algorithm_id}: unknown dataset_set {dataset_set_id!r}")
            continue
        if row.get("task") not in registered_tasks:
            issues.append(f"{algorithm_id}: unknown task {row.get('task')!r}")
        lane = lanes.get(row.get("lane"))
        if lane is None:
            issues.append(f"{algorithm_id}: unknown comparison lane {row.get('lane')!r}")
        elif lane.get("task") != row.get("task"):
            issues.append(f"{algorithm_id}: lane/task mismatch")
        if row.get("current_evidence") not in protocol.get("evidence_tiers", {}):
            issues.append(f"{algorithm_id}: invalid current_evidence")
        used_papers.update(row.get("paper_ids", []))

        valid_for_algorithm = 0
        seen_datasets: set[str] = set()
        for target in dataset_sets[dataset_set_id]:
            family = target.get("dataset_family")
            match = target.get("match")
            fidelity = target.get("protocol")
            if family in seen_datasets:
                issues.append(f"{algorithm_id}: duplicate dataset target {family}")
            seen_datasets.add(family)
            if family not in registered_datasets:
                issues.append(f"{algorithm_id}: unknown dataset family {family}")
            if match not in protocol.get("matching_grades", {}):
                issues.append(f"{algorithm_id}: invalid match grade {match}")
            if fidelity not in protocol.get("protocol_fidelity", {}):
                issues.append(f"{algorithm_id}: invalid protocol fidelity {fidelity}")
            if match == "M1" and target.get("role") != "mismatch_sentinel":
                issues.append(f"{algorithm_id}: M1 target must be mismatch_sentinel")
            pair_count += 1
            if match in {"M2", "M3"}:
                valid_pair_count += 1
                valid_for_algorithm += 1
            if adapter_status.get(family) == "runnable":
                runnable_pair_count += 1
                if match in {"M2", "M3"}:
                    runnable_valid_pair_count += 1
            else:
                pending_targets_by_family[family] = (
                    pending_targets_by_family.get(family, 0) + 1
                )
        if valid_for_algorithm < 3:
            algorithms_below_three_valid_datasets.append(algorithm_id)

    if algorithms_below_three_valid_datasets:
        issues.append(
            "algorithms with fewer than three M2/M3 dataset targets: "
            f"{sorted(algorithms_below_three_valid_datasets)}"
        )

    if len(reference_ids) != len(reference_id_set):
        issues.append("duplicate paper_id in reference experiment backlog")
    if reference_id_set != registered_papers:
        issues.append(
            "reference experiment backlog differs from paper registry: "
            f"missing={sorted(registered_papers - reference_id_set)}; "
            f"extra={sorted(reference_id_set - registered_papers)}"
        )
    if used_papers != registered_papers:
        issues.append(
            "algorithm-paper mappings do not cover paper registry: "
            f"missing={sorted(registered_papers - used_papers)}; "
            f"extra={sorted(used_papers - registered_papers)}"
        )

    matrix_papers_by_algorithm = {
        row["algorithm_id"]: set(row.get("paper_ids", [])) for row in matrix_rows
    }
    for reference in reference_rows:
        paper_id = reference.get("paper_id")
        for algorithm_id in reference.get("algorithm_ids", []):
            if paper_id not in matrix_papers_by_algorithm.get(algorithm_id, set()):
                issues.append(
                    f"reference mapping {paper_id} -> {algorithm_id} is absent from matrix"
                )
        if not reference.get("experiment_items"):
            issues.append(f"{paper_id}: empty experiment_items")
        if not reference.get("status"):
            issues.append(f"{paper_id}: missing status")

    stages = protocol.get("stages", [])
    expected_stage_ids = {
        "S0_preflight",
        "S1_initial_implementation",
        "S2_baseline_tuning",
        "S3_multi_dataset_reproduction",
        "S4_ablation_robustness",
        "S5_held_out_release",
    }
    if {item.get("id") for item in stages} != expected_stage_ids:
        issues.append("protocol stages do not match the required six-stage lifecycle")
    gate_ids = {item.get("id") for item in protocol.get("gates", [])}
    if gate_ids != {
        "G0_validity",
        "G1_activation",
        "G2_multi_dataset",
        "G3_competitive_credit",
        "G4_reference_reproduction",
        "G5_held_out",
    }:
        issues.append("protocol gate set is incomplete")

    status_counts: dict[str, int] = {}
    for row in reference_rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    audit = {
        "harness_id": protocol.get("harness_id"),
        "protocol_status": protocol.get("status"),
        "algorithms": {
            "registered": len(registered_algorithms),
            "book": len(book_ids),
            "sota": len(sota_ids),
            "matrix_covered": len(matrix_id_set),
            "current_E2_or_higher": sum(
                row.get("current_evidence") in {"E2", "E3", "E4", "E5"}
                for row in matrix_rows
            ),
            "three_or_more_valid_dataset_targets": (
                len(matrix_rows) - len(algorithms_below_three_valid_datasets)
            ),
        },
        "datasets": {
            "families": len(registered_datasets),
            "adapter_runnable": sum(value == "runnable" for value in adapter_status.values()),
            "adapter_pending": sum(value != "runnable" for value in adapter_status.values()),
            "pending_targets_by_family": dict(
                sorted(
                    pending_targets_by_family.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "algorithm_dataset_targets": {
            "all": pair_count,
            "M2_M3": valid_pair_count,
            "M1_sentinels": pair_count - valid_pair_count,
            "adapter_runnable_all": runnable_pair_count,
            "adapter_runnable_M2_M3": runnable_valid_pair_count,
            "adapter_pending": pair_count - runnable_pair_count,
        },
        "references": {
            "registered_papers": len(registered_papers),
            "backlog_covered": len(reference_id_set),
            "status_counts": dict(sorted(status_counts.items())),
            "book_chapter_groups": len(references.get("book_chapter_items", [])),
        },
        "issues": issues,
    }
    return audit, issues


def print_plan(audit: dict[str, Any]) -> None:
    targets = audit["algorithm_dataset_targets"]
    print("IIA paper harness execution plan")
    print("01. S0 preflight: finish 11-family priors; implement 6 pending adapters; freeze hashes/splits.")
    print("02. S1 activation: integrate all 30 algorithms into one runner and clear algorithm-specific beacons.")
    print("03. S2 baseline: tune on train/calibration only, at least two datasets, fixed lane parents.")
    print(
        "04. S3 formal: execute "
        f"{targets['M2_M3']} valid algorithm-dataset targets with seeds 1103/2207/3301."
    )
    print("05. S4 robustness: component ablations plus missing/spurious/jitter/delay and prefix/open-set tests.")
    print("06. S5 held-out: once-only score, independent audit, figures, claim ledger, and ARA binding.")
    print(
        f"Startable by adapter state: {targets['adapter_runnable_M2_M3']} valid targets; "
        f"blocked by pending adapters: {targets['adapter_pending']} total targets."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "status", "plan"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    audit, issues = build_audit()
    if args.as_json or args.command == "status":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        print_plan(audit)
    else:
        if issues:
            for issue in issues:
                print(f"[ERROR] {issue}")
        else:
            print(
                "[PASS] "
                f"{audit['algorithms']['matrix_covered']} algorithms; "
                f"{audit['algorithm_dataset_targets']['all']} targets; "
                f"{audit['references']['backlog_covered']} paper rows"
            )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
