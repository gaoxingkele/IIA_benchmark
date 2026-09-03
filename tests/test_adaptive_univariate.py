import numpy as np
import pytest

from iia_benchmark.models import (
    BlockCalibratedECDFAlarm,
    EmpiricalCDFAlarm,
    EmpiricalCDFNormalizer,
    FeatureStabilitySelector,
    RobustMedianScaler,
    RegimeConditionalECDFAlarm,
    SafeRollingECDFAlarm,
)


def test_robust_scaler_round_trip_and_outlier_resistance() -> None:
    values = np.r_[np.linspace(-1.0, 1.0, 101), 1000.0]
    scaler = RobustMedianScaler().fit(values)
    transformed = scaler.transform(values)
    assert abs(np.median(transformed)) < 1e-12
    assert scaler.location_ < 0.02
    np.testing.assert_allclose(scaler.inverse_transform(transformed), values)


def test_empirical_cdf_is_monotone_and_affine_invariant() -> None:
    normal = np.linspace(-3.0, 3.0, 301)
    query = np.linspace(-4.0, 4.0, 401)
    original = EmpiricalCDFNormalizer().fit(normal).transform(query)
    affine = EmpiricalCDFNormalizer().fit(7.0 * normal + 11.0).transform(
        7.0 * query + 11.0
    )
    assert np.all(np.diff(original) >= 0)
    assert original.min() > 0.0 and original.max() < 1.0
    np.testing.assert_allclose(original, affine, atol=1.0 / (len(normal) + 1.0))


def test_empirical_cdf_alarm_has_dimensionless_transfer_and_detects_both_tails() -> None:
    rng = np.random.default_rng(41)
    normal = rng.normal(0.0, 1.0, 4000)
    evaluation = rng.normal(0.0, 1.0, 4000)
    high = rng.normal(5.0, 0.2, 300)
    low = rng.normal(-5.0, 0.2, 300)
    model = EmpiricalCDFAlarm(tail_probability=0.05, tail="two_sided").fit(normal)
    far = float(np.mean(model.predict(evaluation)))
    assert 0.03 < far < 0.08
    assert np.mean(model.predict(high)) > 0.99
    assert np.mean(model.predict(low)) > 0.99

    affine = EmpiricalCDFAlarm(tail_probability=0.05, tail="two_sided").fit(
        9.0 * normal - 4.0
    )
    np.testing.assert_array_equal(
        model.predict(evaluation), affine.predict(9.0 * evaluation - 4.0)
    )


def test_feature_stability_selector_penalizes_direction_reversal() -> None:
    rng = np.random.default_rng(53)
    normal = rng.normal(0.0, 0.2, (900, 2))
    unstable = np.r_[
        rng.normal(3.0, 0.2, 300),
        rng.normal(3.0, 0.2, 300),
        rng.normal(-3.0, 0.2, 300),
    ]
    stable = rng.normal(1.2, 0.2, 900)
    abnormal = np.column_stack([unstable, stable])
    selector = FeatureStabilitySelector().fit(
        normal, abnormal, feature_names=("unstable_large_shift", "stable_shift")
    )
    assert selector.feature_index_ == 1
    assert selector.feature_name_ == "stable_shift"
    assert selector.direction_ == "high"
    np.testing.assert_array_equal(selector.transform(abnormal), stable)
    assert selector.diagnostics_[0].direction_consistency == pytest.approx(2.0 / 3.0)


def test_feature_stability_selector_rejects_bad_shapes_and_constant_matrix() -> None:
    selector = FeatureStabilitySelector()
    with pytest.raises(ValueError, match="two samples and two features"):
        selector.fit(np.ones((4, 1)), np.ones((4, 1)))
    with pytest.raises(ValueError, match="no nonconstant"):
        selector.fit(np.ones((20, 2)), np.ones((20, 2)))


def test_block_calibrator_uses_chronological_validation_and_recent_reference() -> None:
    rng = np.random.default_rng(61)
    normal = np.r_[
        rng.normal(0.0, 0.3, 600),
        rng.normal(0.5, 0.3, 600),
        rng.normal(1.0, 0.3, 600),
    ]
    model = BlockCalibratedECDFAlarm(
        tail="two_sided",
        reference_windows=(128, 256, 512),
        delays=(1, 2, 3),
        block_size=60,
    ).fit(normal)
    assert model.selected_reference_window_ in {128, 256, 512}
    assert model.selected_delay_ in {1, 2, 3}
    assert len(model.calibration_candidates_) == 9
    assert np.isfinite(model.score_samples(normal[-100:])).all()


def test_regime_conditional_alarm_removes_known_operating_level_shift() -> None:
    rng = np.random.default_rng(67)
    first = rng.normal(-5.0, 0.2, 1000)
    second = rng.normal(5.0, 0.2, 1000)
    normal = np.r_[first, second]
    regimes = np.r_[np.zeros(1000, dtype=int), np.ones(1000, dtype=int)]
    model = RegimeConditionalECDFAlarm(tail="two_sided").fit(normal, regimes)
    evaluation = np.r_[
        rng.normal(-5.0, 0.2, 500), rng.normal(5.0, 0.2, 500)
    ]
    evaluation_regimes = np.r_[np.zeros(500, dtype=int), np.ones(500, dtype=int)]
    far = float(np.mean(model.predict(evaluation, evaluation_regimes)))
    assert 0.02 < far < 0.08
    with pytest.raises(ValueError, match="unseen regime"):
        model.predict([0.0, 1.0], [2, 2])


def test_safe_rolling_alarm_updates_central_values_and_freezes_extremes() -> None:
    rng = np.random.default_rng(71)
    normal = rng.normal(0.0, 1.0, 1000)
    model = SafeRollingECDFAlarm(
        tail="two_sided", reference_window=256, update_guard_score=0.80
    ).fit(normal)
    values = np.r_[rng.normal(0.1, 0.8, 300), np.full(30, 20.0)]
    prediction = model.predict(values)
    assert model.last_update_count_ > 200
    assert model.last_frozen_count_ >= 30
    assert np.mean(prediction[-30:]) > 0.95
