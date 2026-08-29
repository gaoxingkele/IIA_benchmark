import numpy as np
import pytest

from iia_benchmark.models import (
    AdaptiveTimeGradient,
    BayesianWindowRegressionAlarm,
    CondenserNOZAlarm,
    CondenserParameters,
    CondenserPhysicalModel,
    SearchConeNOZAlarm,
    VariationDirectionAlarm,
    condenser_alarm_rate_bounds,
    weighted_time_gradient,
)


def test_search_cone_preserves_nonconvex_radial_boundary() -> None:
    rng = np.random.default_rng(3)
    angle = np.r_[rng.normal(0.0, 0.04, 120), rng.normal(np.pi / 2, 0.04, 120)]
    radius = rng.uniform(0.5, 2.0, len(angle))
    normal = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    model = SearchConeNOZAlarm(
        angular_resolution_degrees=12, inference_batch_size=2
    ).fit(normal)
    prediction = model.predict([[1.0, 0.0], [0.0, 1.0], [-4.0, -4.0]])
    np.testing.assert_array_equal(prediction, [0, 0, 1])


def test_search_cone_uses_book_spherical_coordinate_quadrants() -> None:
    model = SearchConeNOZAlarm(angular_resolution_degrees=90)
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    keys = model._keys(directions)
    np.testing.assert_array_equal(keys[:, -1], [0, 0, 1, 3])


def test_weighted_and_adaptive_time_gradient_track_ramp() -> None:
    values = np.arange(30, dtype=float) * 0.25
    assert weighted_time_gradient(values, 0.95) == pytest.approx(0.25)
    rng = np.random.default_rng(5)
    training = rng.normal(0, 0.01, 100)
    extractor = AdaptiveTimeGradient(10, 40).fit(training)
    series = np.r_[np.zeros(50), np.arange(50) * 0.2]
    gradients, directions, scales = extractor.transform(series)
    assert gradients[-1] > 0.15
    assert directions[-1] == 1
    assert np.all((10 <= scales) & (scales <= 40))


def test_variation_direction_rule_matrix_flags_unseen_direction() -> None:
    time = np.arange(200, dtype=float)
    normal = np.column_stack([time * 0.01, time * 0.02])
    model = VariationDirectionAlarm(10, 40).fit(normal)
    abnormal = np.column_stack([time * 0.01, -time * 0.02])
    prediction = model.predict(abnormal)
    assert prediction[-1] == 1


def test_bayesian_window_regression_freezes_abnormal_window() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(200, 2))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(0, 0.02, len(x))
    model = BayesianWindowRegressionAlarm(window_length=10).fit(x, y)
    test_x = rng.normal(size=(40, 2))
    test_y = 1.0 + 2.0 * test_x[:, 0] - test_x[:, 1]
    test_y[20:] += 10.0
    alarms, frozen = model.predict_update(test_x, test_y)
    assert np.mean(alarms[20:]) > 0.8
    assert np.any(frozen[20:])


def test_condenser_equations_are_self_consistent_and_monitor_residual() -> None:
    params = CondenserParameters(2300.0, 10000.0, 10.0, 50.0, 0.02, 0.025)
    model = CondenserPhysicalModel()
    dc = np.linspace(20.0, 30.0, 120)
    t1 = np.linspace(15.0, 22.0, 120)
    t2 = t1 + 8.0
    pressure = model.predict_pressure(dc, t1, t2, params)
    assert np.all(np.isfinite(pressure))
    assert model.goodness_of_fit(pressure, pressure) == pytest.approx(1.0)
    normal = np.column_stack([pressure, dc, t1, t2])
    monitor = CondenserNOZAlarm(params, angular_resolution_degrees=20).fit(normal)
    fault = normal[-1:].copy()
    fault[:, 0] += 20.0
    assert monitor.predict(fault)[0] == 1


def test_condenser_table_3_5_parameters_have_kpa_scale() -> None:
    params = CondenserParameters(2152.7, 24613.0, 14.6996, 12.8538, 0.0182, 0.0247)
    model = CondenserPhysicalModel()
    # The industrial point uses the book's declared process-data unit convention.
    predicted = model.predict_pressure(
        [906.81], [36.08], [44.95], params
    )[0]
    assert 6.0 < predicted < 11.0


def test_condenser_bayesian_far_mar_bounds_count_the_correct_events() -> None:
    bounds = condenser_alarm_rate_bounds(
        [0] * 95 + [1] * 5,
        [1] * 92 + [0] * 8,
        confidence=0.99,
    )
    assert bounds.false_alarm.successes == 5
    assert bounds.false_alarm.trials == 100
    assert bounds.missed_alarm.successes == 8
    assert bounds.missed_alarm.trials == 100
    assert bounds.false_alarm.lower < 0.05 < bounds.false_alarm.upper
    assert bounds.missed_alarm.lower < 0.08 < bounds.missed_alarm.upper
