import json
from pathlib import Path

import numpy as np
import pytest

from iia_benchmark.evaluation import (
    ApplicabilityThresholds,
    assess_univariate_calibration,
    audit_univariate_partitions,
    distribution_shift,
)


def test_distribution_shift_reports_alarm_oriented_effect_sizes() -> None:
    reference = np.linspace(-1.0, 1.0, 101)
    candidate = reference + 2.0
    high = distribution_shift(reference, candidate, direction="high")
    low = distribution_shift(reference, candidate, direction="low")
    assert high.ks > 0.9
    assert high.wasserstein == pytest.approx(2.0)
    assert high.standardized_median_shift > 3.0
    assert low.standardized_median_shift == pytest.approx(
        -high.standardized_median_shift
    )


def test_heldout_audit_detects_baseline_drift_and_direction_reversal() -> None:
    rng = np.random.default_rng(17)
    normal_train = rng.normal(0.0, 0.2, 800)
    normal_evaluation = rng.normal(0.8, 0.2, 800)
    abnormal_calibration = rng.normal(1.5, 0.2, 300)
    abnormal_evaluation = rng.normal(-1.0, 0.2, 300)
    report = audit_univariate_partitions(
        normal_train,
        normal_evaluation,
        abnormal_calibration,
        abnormal_evaluation,
        direction="high",
        threshold=0.6,
    )
    assert report.scope == "held_out_posthoc_diagnostic_not_for_routing"
    assert report.normal_train_to_evaluation.ks > 0.9
    assert report.calibration_auc > 0.99
    assert report.evaluation_auc < 0.01
    assert not report.alarm_direction_consistent
    assert report.threshold_transfer is not None
    assert report.threshold_transfer.normal_evaluation_exceedance_rate > 0.8


def test_calibration_gate_routes_stationary_drift_and_unseparable_cases() -> None:
    rng = np.random.default_rng(23)
    stationary = rng.normal(0.0, 1.0, 1200)
    separable = rng.normal(4.0, 1.0, 600)
    static = assess_univariate_calibration(stationary, separable)
    assert static.status == "static"
    assert static.calibration_auc > 0.99

    drifting_normal = np.r_[
        rng.normal(0.0, 0.1, 600), rng.normal(1.0, 0.1, 600)
    ]
    adapting = assess_univariate_calibration(drifting_normal, separable)
    assert adapting.status == "adapt"
    assert "robust_ecdf" in adapting.recommended_adapters

    weak = rng.normal(0.0, 1.0, 600)
    rejected = assess_univariate_calibration(stationary, weak)
    assert rejected.status == "reject_univariate"
    assert "multivariate_fallback" in rejected.recommended_adapters


def test_calibration_gate_uses_chronological_blocks_for_direction_stability() -> None:
    normal = np.linspace(-0.1, 0.1, 600)
    abnormal = np.r_[
        np.linspace(1.0, 1.2, 200),
        np.linspace(1.0, 1.2, 200),
        np.linspace(-1.2, -1.0, 200),
    ]
    report = assess_univariate_calibration(
        normal,
        abnormal,
        thresholds=ApplicabilityThresholds(minimum_block_auc=0.55),
    )
    assert report.status == "reject_univariate"
    assert report.abnormal_direction_consistency == pytest.approx(2.0 / 3.0)
    assert "insufficient_worst_block_separability" in report.reasons


def test_distribution_audit_rejects_invalid_or_constant_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        distribution_shift(np.ones((2, 2)), np.ones(4))
    with pytest.raises(ValueError, match="constant"):
        distribution_shift(np.ones(10), np.arange(10.0))
    with pytest.raises(ValueError, match="non-finite"):
        distribution_shift(np.r_[np.arange(9.0), np.nan], np.arange(10.0))


def test_registered_real_data_audit_preserves_mechanism_findings() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "experiments/reports/univariate_distribution_audit_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["seeds"] == [1103, 2207, 3301]
    datasets = report["datasets"]
    assert datasets["tep_classic"]["normal_ks"]["median"] < 0.10
    assert datasets["pronto"]["abnormal_ks"]["median"] > 0.50
    assert datasets["pronto"]["normal_lag_one"]["median"] > 0.90
    assert datasets["skab"]["normal_ks"]["median"] > 0.40
    assert (
        datasets["skab"]["raw_threshold_normal_evaluation_exceedance"]["median"]
        > 0.50
    )
    assert "Only calibration_applicability" in report["routing_boundary"]
