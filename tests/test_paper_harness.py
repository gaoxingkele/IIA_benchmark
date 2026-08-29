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
        "current_E2_or_higher": 29,
        "three_or_more_valid_dataset_targets": 29,
        "target_requirement_satisfied": 30,
        "target_requirement_exceptions": 1,
    }
    assert audit["algorithm_dataset_targets"] == {
        "all": 131,
        "M2_M3": 110,
        "M1_sentinels": 21,
        "adapter_runnable_all": 130,
        "adapter_runnable_M2_M3": 109,
        "adapter_pending": 1,
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


def test_chapter4_gap_closure_evidence_is_complete() -> None:
    harness_root = ROOT / "experiments" / "paper_harness" / "chapter4_gap_closure"
    expected_seeds = [1103, 2207, 3301]
    runs = []
    for run_index, expected_seed in enumerate(expected_seeds, start=1):
        run = json.loads(
            (harness_root / f"run_{run_index}" / "final_info.json").read_text(
                encoding="utf-8"
            )
        )
        assert run["seed"] == expected_seed
        assert run["chronological_fold"] == run_index - 1
        assert set(run["results"]) == {
            "controlled_igdte",
            "enas_recursive_bn",
            "piade_plr",
            "imaks_diagnostics",
        }
        assert all(run["mandatory_gates"].values())
        assert run["results"]["controlled_igdte"]["gates"]["mechanism"] is True
        assert run["results"]["enas_recursive_bn"]["prior_gate"]["passed"] is True
        assert run["results"]["piade_plr"]["prior_gate"]["passed"] is True
        assert run["results"]["imaks_diagnostics"]["prior_gate"]["passed"] is True
        runs.append(run)

    assert len({run["config_sha256"] for run in runs}) == 1
    assert len(
        {
            json.dumps(
                run["execution_provenance"]["source_sha256"], sort_keys=True
            )
            for run in runs
        }
    ) == 1

    report = json.loads(
        (
            ROOT
            / "experiments"
            / "reports"
            / "book_ch4_gap_closure_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["run_seeds"] == expected_seeds
    assert report["provenance_gate"]["passed"] is True
    assert report["provenance_gate"]["piade_cross_fold_windows_disjoint"] is True
    assert report["igdte"]["controlled_chain"]["mechanism_passes"] == 3
    assert report["igdte"]["imaks_documented_edge"]["detection_passes"] == 0
    assert report["plr"]["piade"]["chronological_folds"] == [0, 1, 2]
    assert report["strict_paper_score_closure"]["closed"] == 0
    assert report["evidence_updates"] == [
        {"algorithm_id": "book_4_2_igdte", "from": "E0", "to": "E1"},
        {"algorithm_id": "book_4_3_recursive_bn", "from": "E1", "to": "E2"},
        {"algorithm_id": "book_4_4_plr_rca", "from": "E1", "to": "E2"},
    ]
