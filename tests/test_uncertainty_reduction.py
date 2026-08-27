import numpy as np

from iia_benchmark.models import (
    BifurcationDelayTimer,
    BifurcationForecast,
    JackknifePlusRandomForestRegressor,
    UncertaintyReductionForecaster,
    extract_bifurcation_training_data,
)


def _training_data():
    trajectories = []
    for offset in range(6):
        reduction = 2 + offset % 3
        trajectory = []
        for index in range(6):
            trajectory.append(tuple(range(3 if index < reduction else 1)))
        trajectories.append(trajectory)
    return extract_bifurcation_training_data(trajectories, [0, 10, 20, 30, 40, 50])


def test_extract_bifurcation_rows_use_future_reduction_delay() -> None:
    data = _training_data()
    assert data.features.shape[1] == 4
    assert np.all(data.time_to_reduction > 0)
    assert set(data.time_to_reduction) <= {10.0, 20.0, 30.0, 40.0}


def test_jackknife_plus_interval_is_finite_ordered_and_reproducible() -> None:
    data = _training_data()
    first = JackknifePlusRandomForestRegressor(n_estimators=8, random_state=4).fit(
        data.features, data.time_to_reduction
    )
    second = JackknifePlusRandomForestRegressor(n_estimators=8, random_state=4).fit(
        data.features, data.time_to_reduction
    )
    point, interval = first.predict(data.features[:3], error_rate=0.2)
    point_again, interval_again = second.predict(data.features[:3], error_rate=0.2)
    np.testing.assert_allclose(point, point_again)
    np.testing.assert_allclose(interval, interval_again)
    assert np.all(interval[:, 0] <= interval[:, 1])
    assert np.all(np.isfinite(interval))


def test_forecaster_returns_typed_interval() -> None:
    data = _training_data()
    model = UncertaintyReductionForecaster(n_estimators=6, random_state=2).fit(data)
    forecast = model.forecast(data.features[0], error_rate=0.2)
    assert forecast.lower <= forecast.upper
    assert forecast.point >= 0


def test_delay_timer_rejects_unstable_and_emits_stable_forecasts() -> None:
    timer = BifurcationDelayTimer(delay=3, tolerance=1.0)
    assert timer.update(BifurcationForecast(10, 8, 12, 0.1)) is None
    assert timer.update(BifurcationForecast(20, 18, 22, 0.1)) is None
    assert timer.update(BifurcationForecast(20.5, 18, 23, 0.1)) is None
    result = timer.update(BifurcationForecast(20.2, 19, 22, 0.1))
    assert result is not None
    assert result.point == 20.2
    assert result.lower == 18
    assert result.upper == 23
