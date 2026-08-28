import json
from pathlib import Path

import pytest

from iia_benchmark.runner import run_experiment


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "synthetic_univariate_smoke.json",
        "synthetic_flood_similarity_smoke.json",
        "synthetic_multivariate_noz_smoke.json",
        "synthetic_root_cause_smoke.json",
    ],
)
def test_smoke_experiments(name: str) -> None:
    config_path = ROOT / "configs" / "experiments" / name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_experiment(config_path)
    assert result["experiment_id"]
    assert (ROOT / config["outputs"]["run_dir"] / "result.json").exists()


def test_book_manifest() -> None:
    manifest = json.loads(
        (ROOT / "papers" / "extracted_text" / "book" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["pdf_pages"] == 433
    assert [(chapter["pdf_start"], chapter["pdf_end"]) for chapter in manifest["chapters"]] == [
        (13, 59),
        (60, 138),
        (139, 230),
        (231, 311),
        (312, 388),
        (389, 428),
    ]


def test_pronto_representation_ablation_reports_preserve_degenerate_results() -> None:
    reports = ROOT / "experiments" / "reports"
    ctfh_state = json.loads(
        (reports / "pronto_ctfh_fault_classification_validation.json").read_text(
            encoding="utf-8"
        )
    )["result"]
    ctfh_edge = json.loads(
        (reports / "pronto_ctfh_activation_classification_validation.json").read_text(
            encoding="utf-8"
        )
    )["result"]
    assert ctfh_state["alarm_representation"] == "state"
    assert ctfh_edge["alarm_representation"] == "rising_edge"
    assert ctfh_state["metrics"] == ctfh_edge["metrics"]
    assert all(
        profile["consensus_hashes"] == 0
        for profile in ctfh_state["model_diagnostics"]["profiles"]
    )

    for name, representation in (
        ("pronto_cone_uncertainty_validation.json", "state"),
        ("pronto_cone_activation_uncertainty_validation.json", "rising_edge"),
        ("pronto_cross_conformal_uncertainty_validation.json", "state"),
        (
            "pronto_cross_conformal_activation_uncertainty_validation.json",
            "rising_edge",
        ),
    ):
        result = json.loads((reports / name).read_text(encoding="utf-8"))["result"]
        assert result["alarm_representation"] == representation
        assert result["metrics"]["coverage"] == 1.0
        assert result["metrics"]["average_set_size"] == 4.0
        assert result["metrics"]["singleton_accuracy"] is None
