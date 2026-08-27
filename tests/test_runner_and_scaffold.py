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
