from __future__ import annotations

import json
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CARD_ROOT = ROOT / "paper_harness" / "paper_exact"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_paper_grid_module():
    path = ROOT / "experiments/paper_harness/p0_paper_exact/paper_grid.py"
    spec = importlib.util.spec_from_file_location("p0_paper_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0_protocol_cards_and_capsule_manifest_are_closed() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/paper_exact.py", "check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    audit = json.loads(completed.stdout)
    assert audit["paper_exact_cards"] == 3
    assert audit["issues"] == []

    manifest = load(ROOT / "configs/reproducibility/codeocean_capsules.v1.json")
    assert {row["paper_id"] for row in manifest["capsules"]} == {
        "faulwasser2024_casim",
        "faulwasser2024_cone_afc",
        "faulwasser2025_uncertainty_reduction",
    }
    assert all(row["acquisition_status"] == "complete_verified" for row in manifest["capsules"])
    assert all(row["code_license"] == "MIT" for row in manifest["capsules"])
    assert all(row["data_license"] == "CC0-1.0" for row in manifest["capsules"])

    experiment = load(ROOT / "configs/experiments/p0_paper_exact.json")
    assert [row["id"] for row in experiment["passes"]] == [
        "author_capsule_default",
        "author_code_paper_grid",
        "independent_same_fold",
    ]
    assert experiment["paper_grid"]["faulwasser2024_casim"]["train_test_sets"] == 70
    assert experiment["paper_grid"]["faulwasser2024_casim"]["random_instances"] == 10
    assert experiment["paper_grid"]["faulwasser2024_casim"]["total_model_fit_tasks"] == 700
    assert experiment["paper_grid"]["faulwasser2024_cone_afc"][
        "calibration_per_class"
    ] == [22, 102, 2491]
    assert experiment["paper_grid"]["faulwasser2024_cone_afc"]["model_split_tasks"] == 250
    assert experiment["paper_grid"]["faulwasser2024_cone_afc"]["checkpoint_unit"] == "split_and_model"
    assert experiment["acceptance"]["docker_image_run_required_for_P3"] is True


def test_protocol_cards_freeze_required_experiment_details() -> None:
    casim = load(CARD_ROOT / "faulwasser2024_casim.v1.json")
    cone = load(CARD_ROOT / "faulwasser2024_cone_afc.v1.json")
    bip = load(CARD_ROOT / "faulwasser2025_uncertainty_reduction.v1.json")

    assert casim["split_protocol"]["open_set_train_test_sets"] == 70
    assert casim["randomness"]["multirocket_repetitions_in_paper"] == 10
    assert casim["hyperparameters"]["casim"]["num_features"] == 672
    assert len(casim["paper_targets"]) == 4

    assert cone["data_protocol"]["samples"] == 18750
    assert cone["split_protocol"]["paper_tests"] == 50
    assert cone["split_protocol"]["calibration_samples_per_class"] == [22, 102, 2491]
    assert len(cone["paper_targets"]) == 19
    assert set(cone["reference_tables"]["Table_1_accuracy_and_coverage"]) == {
        "WDI_1NN",
        "ACM_SVM",
        "CASIM",
        "EAC_1NN",
        "MBW_LR",
    }
    assert all(
        len(row["coverage"]) == 9
        for row in cone["reference_tables"]["Table_1_accuracy_and_coverage"].values()
    )
    assert all(
        len(row) == 9
        for row in cone["reference_tables"]["Table_2_average_set_size"].values()
    )

    assert bip["data_protocol"]["synthetic"]["samples"] == 1875
    assert bip["data_protocol"]["tep"]["samples"] == 1000
    assert len(bip["paper_targets"]) == 16
    assert any("overlap" in item for item in bip["known_mismatches"])


def test_casim_author_capsule_default_result_is_retained() -> None:
    result = load(
        ROOT
        / "experiments/paper_harness/p0_paper_exact/run_1/final_info.json"
    )
    assert result["paper_id"] == "faulwasser2024_casim"
    assert result["author_code_unchanged"] is True
    assert result["reproduction_level"] == "P2_author_capsule_default"
    assert result["paper_exact_closed"] is False
    assert len(result["metrics"]["fold_balanced_accuracy"]) == 5
    assert result["metrics"]["mean_balanced_accuracy"] > 0.99
    sensitivity = result["metrics"]["duplicate_excluded_test_sensitivity"]
    assert sensitivity["excluded_test_instances"] == 4
    assert sensitivity["mean_duplicate_excluded_balanced_accuracy"] > 0.99
    environment = load(
        ROOT / "experiments/paper_harness/p0_paper_exact/run_1/environment.json"
    )
    assert environment["exact_dependency_match"] is True
    assert environment["native_compatibility_run"] is True
    assert environment["authoritative_engine"] == "docker"


def test_cone_author_capsule_default_result_is_retained_separately() -> None:
    result = load(
        ROOT
        / "experiments/paper_harness/p0_paper_exact/run_2/final_info.json"
    )
    assert result["paper_id"] == "faulwasser2024_cone_afc"
    assert result["author_code_unchanged"] is True
    assert result["reproduction_level"] == "P2_author_capsule_default"
    assert result["paper_exact_closed"] is False
    assert set(result["metrics"]["models"]) == {
        "WDI_1NN",
        "ACM_SVM",
        "CASIM",
        "EAC_1NN",
        "MBW_LR",
    }
    assert len(result["paper_comparison"]) == 10
    assert sum(row["numeric_within_tolerance"] for row in result["paper_comparison"]) == 9
    assert all(row["protocol_match"] is False for row in result["paper_comparison"])
    assert result["metrics"]["models"]["CASIM"]["mean_coverage_all_prefixes"] > 0.9

    usage = load(
        ROOT / "experiments/paper_harness/p0_paper_exact/run_2/resource_usage.json"
    )
    assert usage["return_code"] == 0
    assert usage["wall_time_seconds"] > 0


def test_p0_data_priors_are_frozen_before_model_comparison() -> None:
    report = load(ROOT / "experiments/reports/p0_codeocean_data_prior.json")
    assert report["all_prior_gates_passed"] is True
    datasets = report["datasets"]
    assert datasets["cone_synthetic"]["canonical_unique_trajectories"] == 18750
    assert datasets["bip_synthetic"]["canonical_unique_trajectories"] == 1875
    assert datasets["bip_tep"]["canonical_unique_trajectories"] == 1000
    assert datasets["bip_tep"]["channel_counts"] == {"50": 1000}
    assert datasets["casim_tep"]["canonical_unique_trajectories"] == 308
    assert datasets["casim_tep"]["duplicate_episodes_beyond_first"] == 2
    assert datasets["casim_tep"]["official_split_duplicate_audit"][
        "folds_with_duplicate_leakage"
    ] == [3, 4, 5]
    assert datasets["casim_tep"]["official_split_duplicate_audit"][
        "crossing_group_instances"
    ] == 4
    assert all(
        len(group["members"]) == 2
        and len({member["label"] for member in group["members"]}) == 1
        for group in datasets["casim_tep"]["duplicate_groups"]
    )


def test_p0_pause_checkpoint_is_explicit_and_resumable() -> None:
    checkpoint = load(
        ROOT
        / "experiments/paper_harness/p0_paper_exact/checkpoint_2026-08-30.json"
    )
    assert checkpoint["active_experiment_processes_after_pause"] == 0
    rows = checkpoint["experiments"]
    assert rows["faulwasser2024_casim"]["completed_tasks"] == 48
    assert rows["faulwasser2024_casim"]["total_tasks"] == 70
    assert rows["faulwasser2024_cone_afc"]["completed_full_splits"] == 0
    assert rows["faulwasser2024_cone_afc"]["smoke_result"][
        "paper_grid_evidence"
    ] is False
    assert rows["faulwasser2025_uncertainty_reduction"][
        "completed_dataset_model_groups"
    ] == ["tep/MBW_LR", "tep/EAC_1NN"]
    assert rows["faulwasser2025_uncertainty_reduction"]["limitations"]
    assert (ROOT / "scripts/resume_p0_checkpoint.ps1").is_file()


def test_casim_open_set_partial_grid_is_seeded_and_not_mislabeled_complete() -> None:
    root = ROOT / "experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_1"
    summary = load(root / "summary.json")
    rows = [
        json.loads(line)
        for line in (root / "seed_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 70
    assert len({row["task_id"] for row in rows}) == 70
    assert {(row["held_out_class"], row["fold"]) for row in rows} == {
        (held_out, fold) for held_out in range(14) for fold in range(5)
    }
    assert {row["random_seed"] for row in rows} == {42}
    assert summary["complete_requested_grid"] is True
    assert summary["paper_required_repetitions"] == 10
    assert summary["complete_paper_grid"] is False


def test_casim_open_set_equality_boundary_is_rejected_as_novel() -> None:
    module = load_paper_grid_module()
    result = module.summarize_casim_open_set(
        [
            {
                "y_true": [0, -1],
                "point_prediction": [0, 0],
                "novelty_score": [0.0, 0.001],
            }
        ],
        repetitions=1,
    )
    assert result["mean_TPR"][0] == 1.0
    assert result["mean_TNR"][0] == 1.0
    assert result["mean_balanced_accuracy"][0] == 1.0


def test_casim_open_set_full_random_grid_and_envelope_are_complete() -> None:
    root = ROOT / "experiments/paper_harness/p0_paper_exact/run_1/paper_grid/repetitions_10"
    summary = load(root / "summary.json")
    rows = [
        json.loads(line)
        for line in (root / "seed_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 700
    assert {row["task_id"] for row in rows} == set(range(700))
    assert {row["random_seed"] for row in rows} == set(range(42, 52))
    assert all(
        sum(row["random_seed"] == seed for row in rows) == 70
        for seed in range(42, 52)
    )
    assert summary["complete_requested_grid"] is True
    assert summary["complete_paper_grid"] is True
    assert len(summary["random_instance_summaries"]) == 10
    envelope = summary["balanced_accuracy_random_instance_envelope"]
    assert all(len(envelope[key]) == 1000 for key in ("minimum", "q25", "q75", "maximum"))
    assert summary["maximum"]["balanced_accuracy"] > 0.94
    assert abs(summary["paper_deltas"]["maximum_balanced_accuracy"]) <= 0.02
    assert abs(summary["paper_deltas"]["mean_balanced_accuracy_over_thresholds"]) > 0.02

    status = load(CARD_ROOT / "status.v1.json")["papers"][0]["paper_exact_result"]
    assert status["open_set"]["completed_model_fit_tasks"] == 700
    assert status["open_set"]["complete_paper_grid"] is True
    assert status["P3_closed"] is False
    assert hashlib.sha256((root / "seed_results.jsonl").read_bytes()).hexdigest() == summary[
        "seed_results_sha256"
    ]


def test_cone_model_split_checkpoint_is_granular_and_hash_stable() -> None:
    root = ROOT / "experiments/paper_harness/p0_paper_exact/run_2/paper_grid"
    tasks = [
        json.loads(line)
        for line in (root / "model_split_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert tasks
    assert len({row["task_id"] for row in tasks}) == len(tasks)
    assert all(len(row["models"]) == 1 for row in tasks)
    assert any(row["task_id"] == "split=0|model=WDI_1NN" for row in tasks)
    summary = load(root / "summary.json")
    assert summary["model_split_tasks_completed"] == 1
    assert summary["model_split_tasks_required"] == 250
    assert summary["complete_paper_grid"] is False
    assert hashlib.sha256((root / "model_split_results.jsonl").read_bytes()).hexdigest() == summary[
        "task_checkpoint_sha256"
    ]
