import numpy as np
import pytest

from iia_benchmark.evaluation import (
    alarm_event_metrics,
    block_event_rate_posterior,
    block_bootstrap_alarm_metrics,
)


def test_event_metrics_count_runs_rate_and_detection_delay() -> None:
    normal = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=int)
    abnormal = np.array([0, 0, 1, 1, 1], dtype=int)
    result = alarm_event_metrics(normal, abnormal, sample_period_seconds=900.0)
    assert result.false_alarm_events == 2
    assert result.false_alarm_events_per_hour == pytest.approx(1.0)
    assert result.normal_alarm_mean_duration_samples == pytest.approx(1.5)
    assert result.normal_alarm_maximum_duration_samples == 2
    assert result.abnormal_event_recall == 1.0
    assert result.detection_delay_samples == 2
    assert result.detection_delay_seconds == 1800.0


def test_block_bootstrap_is_reproducible_and_preserves_point_estimate() -> None:
    normal = np.tile([0, 0, 0, 1, 1, 0], 100)
    abnormal = np.tile([0, 1, 1, 1, 1, 0], 80)
    first = block_bootstrap_alarm_metrics(
        normal, abnormal, block_size=12, draws=100, confidence=0.90, seed=83
    )
    second = block_bootstrap_alarm_metrics(
        normal, abnormal, block_size=12, draws=100, confidence=0.90, seed=83
    )
    assert first.as_dict() == second.as_dict()
    assert first.metrics["false_alarm_rate"].point_estimate == pytest.approx(1.0 / 3.0)
    assert first.metrics["missed_alarm_rate"].point_estimate == pytest.approx(1.0 / 3.0)
    assert first.metrics["f1"].lower <= first.metrics["f1"].point_estimate <= first.metrics["f1"].upper


def test_event_uncertainty_validates_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        alarm_event_metrics([0, 1], [1, 1], sample_period_seconds=0.0)
    with pytest.raises(ValueError, match="draws"):
        block_bootstrap_alarm_metrics([0, 1], [1, 1], draws=10)


def test_block_event_posterior_is_finite_for_zero_events() -> None:
    posterior = block_event_rate_posterior(
        np.zeros(120, dtype=int), block_size=30, confidence=0.90
    )
    assert posterior.events == 0
    assert posterior.blocks == 4
    assert posterior.posterior_mean == pytest.approx(1.0 / 6.0)
    assert 0.0 < posterior.lower < posterior.upper < 1.0
