from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_paper_harness_matrix_is_closed() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/paper_harness.py", "status"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    audit = json.loads(completed.stdout)
    assert audit["issues"] == []
    assert audit["algorithms"] == {
        "registered": 30,
        "book": 20,
        "sota": 10,
        "matrix_covered": 30,
        "current_E2_or_higher": 27,
        "three_or_more_valid_dataset_targets": 29,
        "target_requirement_satisfied": 30,
        "target_requirement_exceptions": 1,
    }
    assert audit["algorithm_dataset_targets"] == {
        "all": 126,
        "M2_M3": 105,
        "M1_sentinels": 21,
        "adapter_runnable_all": 123,
        "adapter_runnable_M2_M3": 102,
        "adapter_pending": 3,
    }
    assert audit["references"]["registered_papers"] == 28
    assert audit["references"]["backlog_covered"] == 28


def test_sota_wave2_multidataset_evidence_is_complete() -> None:
    harness_root = ROOT / "experiments" / "paper_harness" / "sota_wave2_multidataset"
    expected_seeds = [1103, 2207, 3301]
    expected_datasets = {
        "tep_alarm_dataport",
        "npp_alarm_dataport",
        "fcc_alarm",
    }
    expected_models = {
        "jaccard_class_core",
        "ctfh_fingerprinting",
        "structured_hdam",
        "casim",
        "modified_tfidf_afc",
        "time_encoded_histogram_hybrid",
    }

    runs = []
    for run_index, expected_seed in enumerate(expected_seeds, start=1):
        run = json.loads(
            (harness_root / f"run_{run_index}" / "final_info.json").read_text(
                encoding="utf-8"
            )
        )
        assert run["seed"] == expected_seed
        assert set(run["results"]) == expected_datasets
        assert all(result["prior_gate"]["passed"] for result in run["results"].values())
        assert all(
            set(result["classification"]) == expected_models
            for result in run["results"].values()
        )
        runs.append(run)

    assert len({run["config_sha256"] for run in runs}) == 1
    report = json.loads(
        (
            ROOT
            / "experiments"
            / "reports"
            / "sota_wave2_multidataset_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["run_seeds"] == expected_seeds
    assert report["provenance_gate"]["passed"] is True
    assert report["strict_paper_score_closure"] == {
        "closed": 0,
        "total": 9,
        "reason": "Exact paper data/splits/full text or official capsules remain unavailable.",
    }
    assert {update["algorithm_id"] for update in report["evidence_updates"]} == {
        "modified_tfidf_afc_2025",
        "hybrid_histogram_afc_2026",
        "uncertainty_reduction_2025",
        "etfa_robustness_2025",
        "afc_robustbench_2026",
    }
