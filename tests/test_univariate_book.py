import numpy as np
import pytest

from iia_benchmark.models import (
    AlarmOnOffDelay,
    alarm_episode_metrics,
    bayesian_duration_tail,
    beta_binomial_posterior,
    binary_run_lengths,
    build_alarm_probability_plot,
    deadband_index,
    design_deadband_width,
    design_iid_delay_timer,
    design_non_iid_delay_timer,
    iid_delay_timer_performance,
    pettitt_test,
    recursive_pettitt_segments,
    select_alarm_probability_threshold,
)


def test_iid_delay_equations_reduce_to_basic_alarm_at_n_one() -> None:
    result = iid_delay_timer_performance(0.02, 0.9, 1, sample_period=2.0)
    assert result.false_alarm_rate == pytest.approx(0.02)
    assert result.missed_alarm_rate == pytest.approx(0.1)
    assert result.average_alarm_delay == pytest.approx(2.0 * 0.1 / 0.9)
    np.testing.assert_array_equal(
        AlarmOnOffDelay(1.0, delay=1).predict([0.0, 1.0, 2.0, 0.0]),
        [0, 1, 1, 0],
    )


def test_symmetric_alarm_on_off_delay_requires_n_samples_both_ways() -> None:
    prediction = AlarmOnOffDelay(1.0, delay=2).predict(
        [0.0, 1.1, 0.9, 1.1, 1.2, 0.8, 1.1, 0.7, 0.6]
    )
    np.testing.assert_array_equal(prediction, [0, 0, 0, 0, 1, 1, 1, 1, 0])


def test_iid_delay_reduces_far_and_increases_delay() -> None:
    basic = iid_delay_timer_performance(0.05, 0.8, 1)
    delayed = iid_delay_timer_performance(0.05, 0.8, 3)
    assert delayed.false_alarm_rate < basic.false_alarm_rate
    assert delayed.average_alarm_delay > basic.average_alarm_delay


def test_iid_joint_design_and_xu2012_table_vii() -> None:
    table = {
        2: (0.0468, 0.0305, 1.4294),
        3: (0.0116, 0.0060, 2.8988),
        4: (0.0025, 0.0010, 4.5694),
    }
    for delay, expected in table.items():
        result = iid_delay_timer_performance(0.1486, 1.0 - 0.1204, delay)
        np.testing.assert_allclose(
            (
                result.false_alarm_rate,
                result.missed_alarm_rate,
                result.average_alarm_delay,
            ),
            expected,
            atol=1e-4,
        )
    normal = np.tile([0.0, 0.0, 0.0, 2.0], 100)
    abnormal = np.tile([0.0, 2.0, 2.0, 2.0], 100)
    design = design_iid_delay_timer(
        normal,
        abnormal,
        thresholds=[0.5, 1.0, 1.5],
        delays=[1, 2, 3],
    )
    assert design.threshold in {0.5, 1.0, 1.5}
    assert design.delay in {1, 2, 3}
    assert np.isfinite(design.loss)


def test_pettitt_detects_and_recursively_segments_mean_shift() -> None:
    rng = np.random.default_rng(7)
    values = np.r_[rng.normal(0, 0.15, 80), rng.normal(2, 0.15, 80)]
    result = pettitt_test(values)
    assert abs(result.change_index - 80) <= 2
    assert result.p_value < 0.001
    assert recursive_pettitt_segments(values, min_size=30) == [(0, 80), (80, 160)]


def test_runs_and_bayesian_tail_probability() -> None:
    runs = binary_run_lengths([0, 1, 1, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(runs, [2, 3])
    posterior = bayesian_duration_tail([1, 2, 3, 4], 3)
    assert posterior.mean == pytest.approx(0.5)
    assert posterior.lower < posterior.mean < posterior.upper
    assert beta_binomial_posterior(0, 0).mean == pytest.approx(0.5)


def test_non_iid_delay_design_is_finite_and_reproducible() -> None:
    normal = np.tile([0.0, 1.2, 1.3, 0.0, 0.0], 30)
    abnormal = np.tile([1.4, 1.5, 0.6, 1.6, 1.7], 30)
    result = design_non_iid_delay_timer(
        normal,
        abnormal,
        thresholds=[1.0, 1.25],
        delays=[1, 2, 3],
    )
    assert result.threshold in {1.0, 1.25}
    assert result.delay in {1, 2, 3}
    assert np.isfinite(result.loss)


def test_non_iid_delay_design_retains_zero_event_uncertainty() -> None:
    result = design_non_iid_delay_timer(
        np.zeros(200),
        np.ones(200),
        thresholds=[0.5],
        delays=[1, 2],
    )
    assert result.zero_event_fallback
    assert result.normal_alarm_runs == 0
    assert result.abnormal_no_alarm_runs == 0
    assert 0 < result.false_alarm.upper < 0.05
    assert 0 < result.missed_alarm.upper < 0.05


def test_deadband_episode_index_and_width_design() -> None:
    values = np.tile([0.0, 1.01, 1.02, 1.01, 1.02, 0.0], 20)
    episodes = alarm_episode_metrics(values, 1.0)
    assert len(episodes.durations) == 20
    np.testing.assert_array_equal(episodes.durations, np.full(20, 4))
    index = deadband_index(values, 1.0, duration_max=10.0, deviation_max=0.2)
    assert index.angle_degrees > 45
    assert index.suitable
    design = design_deadband_width(
        episodes.deviations,
        maximum_width=0.2,
        target_remaining_probability=0.05,
    )
    assert 0.0 <= design.width <= 0.2
    assert design.remaining_probability.mean < 0.1


def test_alarm_probability_plot_quantization_and_threshold_selection() -> None:
    pattern = np.array([0.0, 0.4, 0.8, 1.1, 1.4, 1.8, 1.4, 1.1, 0.8, 0.4])
    values = np.tile(pattern, 80)
    plot = build_alarm_probability_plot(values, 1.0, minimum_state_samples=80)
    assert plot.first_alarm_state >= 2
    assert plot.transition_matrix.shape[0] == len(plot.boundaries) - 1
    np.testing.assert_allclose(plot.transition_matrix.sum(axis=1), 1.0)
    assert plot.p_to_alarm
    assert plot.p_after_alarm
    result = select_alarm_probability_threshold(
        values, [0.9, 1.0, 1.2], minimum_state_samples=80
    )
    assert result.threshold in {0.9, 1.0, 1.2}
    assert np.isfinite(result.score)
