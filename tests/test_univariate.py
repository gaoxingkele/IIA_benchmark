import numpy as np

from iia_benchmark.data import make_synthetic_alarm_run
from iia_benchmark.models import ThresholdDelayDeadband, design_alarm


def test_delay_and_deadband_state_machine() -> None:
    model = ThresholdDelayDeadband(threshold=2.0, delay=2, deadband=0.5)
    alarm = model.predict([1.0, 2.1, 2.2, 1.8, 1.4, 2.2, 1.0])
    np.testing.assert_array_equal(alarm, [0, 0, 1, 1, 0, 0, 0])


def test_book_style_grid_search_returns_valid_model() -> None:
    run = make_synthetic_alarm_run(length=500, change_at=300)
    result = design_alarm(
        run.values[:, 0],
        run.abnormal,
        thresholds=np.linspace(0.8, 2.5, 8),
        delays=(1, 3),
        deadbands=(0.0, 0.2),
    )
    assert np.isfinite(result.loss)
    assert 0 <= result.false_alarm_rate <= 1
    assert 0 <= result.missed_alarm_rate <= 1
