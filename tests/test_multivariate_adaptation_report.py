import json
from pathlib import Path

import pytest


def test_multivariate_adaptation_report_matches_frozen_chapter3_baseline() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "experiments/reports/multivariate_distribution_adaptation_validation.json"
        ).read_text(encoding="utf-8")
    )
    chapter3 = json.loads(
        (root / "experiments/reports/book_ch3_multidataset_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["seeds"] == [1103, 2207, 3301]
    assert report["stage_acceptance"]["same_chapter3_protocol"]
    assert report["stage_acceptance"]["all_uncertainty_present"]
    for dataset in ("tep_classic", "pronto", "skab"):
        observed = report["datasets"][dataset]["M0"]["metrics"]
        expected = chapter3["aggregate_metrics"][dataset]["baseline_mahalanobis"]
        assert observed["f1"]["mean"] == pytest.approx(expected["f1"])
        assert observed["false_alarm_rate"]["mean"] == pytest.approx(expected["far"])
        assert observed["missed_alarm_rate"]["mean"] == pytest.approx(expected["mar"])


def test_multivariate_adaptation_report_retains_negative_transfer_and_denials() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "experiments/reports/multivariate_distribution_adaptation_validation.json"
        ).read_text(encoding="utf-8")
    )
    datasets = report["datasets"]
    for dataset in ("tep_classic", "pronto", "skab"):
        baseline = datasets[dataset]["M0"]["metrics"]["f1"]["mean"]
        adapted = datasets[dataset]["M2"]["metrics"]["f1"]["mean"]
        assert adapted < baseline
    assert datasets["tep_classic"]["M3"]["denied_units"] == 3
    assert datasets["tep_classic"]["M3"]["coverage"] == pytest.approx(2.0 / 3.0)
    assert datasets["skab"]["M2"]["metrics"]["false_alarm_rate"]["mean"] > 0.9
