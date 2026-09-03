import numpy as np
import pytest

from iia_benchmark.evaluation import (
    assess_multivariate_calibration,
    multivariate_distribution_shift,
)
from iia_benchmark.models import (
    AdaptiveMultivariateAlarmRouter,
    BlockCalibratedRobustMahalanobisAlarm,
    RobustShrinkageMahalanobisAlarm,
)


def test_robust_shrinkage_is_affine_invariant_and_handles_collinearity() -> None:
    rng = np.random.default_rng(101)
    base = rng.normal(size=(1000, 2))
    normal = np.column_stack([base[:, 0], base[:, 1], base[:, 0] + base[:, 1]])
    evaluation = normal[:200] + rng.normal(0.0, 0.02, (200, 3))
    model = RobustShrinkageMahalanobisAlarm(shrinkage=0.2).fit(normal)
    transformed = RobustShrinkageMahalanobisAlarm(shrinkage=0.2).fit(
        normal * np.array([2.0, 5.0, 9.0]) + np.array([3.0, -8.0, 11.0])
    )
    np.testing.assert_allclose(
        model.score_samples(evaluation),
        transformed.score_samples(
            evaluation * np.array([2.0, 5.0, 9.0]) + np.array([3.0, -8.0, 11.0])
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isfinite(model.covariance_condition_)


def test_block_calibrated_multivariate_alarm_selects_registered_candidate() -> None:
    rng = np.random.default_rng(103)
    normal = np.r_[
        rng.normal(0.0, 1.0, (600, 3)),
        rng.normal(0.4, 1.0, (600, 3)),
    ]
    model = BlockCalibratedRobustMahalanobisAlarm(
        reference_windows=(128, 256, 512), delays=(1, 2, 3), block_size=30
    ).fit(normal)
    assert model.selected_reference_window_ in {128, 256, 512}
    assert model.selected_delay_ in {1, 2, 3}
    assert len(model.calibration_candidates_) == 9
    assert np.isfinite(model.score_samples(normal[-100:])).all()


def test_multivariate_distribution_audit_detects_covariance_and_temporal_shift() -> None:
    rng = np.random.default_rng(107)
    reference = rng.normal(size=(1000, 3))
    candidate = np.empty((1000, 3))
    candidate[0] = rng.normal(size=3)
    for index in range(1, len(candidate)):
        innovation = rng.multivariate_normal(
            np.zeros(3), np.full((3, 3), 0.7) + np.eye(3) * 0.3
        )
        candidate[index] = 0.95 * candidate[index - 1] + innovation
    audit = multivariate_distribution_shift(reference, candidate)
    assert audit.covariance_relative_frobenius_shift > 1.0
    assert audit.maximum_absolute_correlation_shift > 0.3
    assert audit.candidate_absolute_lag_one_median > 0.8


def test_multivariate_router_scores_stable_shift_and_denies_weak_blocks() -> None:
    rng = np.random.default_rng(109)
    normal = rng.normal(0.0, 1.0, (1200, 3))
    abnormal = rng.normal(4.0, 1.0, (600, 3))
    applicability = assess_multivariate_calibration(normal, abnormal)
    assert applicability.status in {"static", "adapt"}
    router = AdaptiveMultivariateAlarmRouter().fit(normal, abnormal)
    assert np.mean(router.predict(abnormal)) > 0.8

    weak = rng.normal(0.0, 1.0, (600, 3))
    denied = AdaptiveMultivariateAlarmRouter().fit(normal, weak)
    assert denied.decision_.status == "reject_multivariate"
    with pytest.raises(RuntimeError, match="denied"):
        denied.predict(weak)
