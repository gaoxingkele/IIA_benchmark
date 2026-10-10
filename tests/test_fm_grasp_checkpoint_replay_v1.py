import copy

import numpy as np
import pytest

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import compare_full_scores, compare_metrics, verify_checkpoint_metadata


def test_full_scores_require_identical_coverage_and_finite_values():
    expected = np.array([10., 20., 30.])
    tolerance = dict(rtol=2e-4, atol=1e-3)
    assert compare_full_scores(expected.copy(), expected, tolerance)['bitwise_identical']
    assert not compare_full_scores(expected + 1e-5, expected, tolerance)['bitwise_identical']
    for actual in (expected[:2], np.array([10., 20., np.nan]), expected + 1):
        with pytest.raises(ValueError):
            compare_full_scores(actual, expected, tolerance)


def test_metric_agreement_cannot_promote_test_calibration_or_pa():
    expected = dict(validation_only_control=dict(f1=.3, threshold=100., quantile=.99, point_adjustment=False))
    tolerance = dict(metric_atol=2e-4, threshold_rtol=2e-4, threshold_atol=1e-3)
    assert compare_metrics(expected, expected, tolerance)['validation_only_control/f1'] == 0
    for field, value in [('point_adjustment', True), ('quantile', .95), ('f1', .31)]:
        actual = copy.deepcopy(expected)
        actual['validation_only_control'][field] = value
        with pytest.raises(ValueError):
            compare_metrics(actual, expected, tolerance)


def test_checkpoint_metadata_rejects_another_seed_model_and_partial_training():
    result = dict(job=dict(seed=1103, expected_optimizer_updates=93000), job_sha256='binding',
                  data_audit=dict(test_points=72000), epochs_completed=1500, optimizer_updates=93000)
    checkpoint = dict(job_sha256='binding', seed=1103, parameters=dict(hidden=128),
                      entity_data_audit=result['data_audit'], epochs_completed=1500, optimizer_updates=93000)
    verify_checkpoint_metadata(checkpoint, result, dict(hidden=128), lambda _: 'binding')
    for field, value in [('seed', 1104), ('epochs_completed', 1499), ('optimizer_updates', 92999), ('parameters', dict(hidden=64))]:
        bad = copy.deepcopy(checkpoint)
        bad[field] = value
        with pytest.raises(ValueError):
            verify_checkpoint_metadata(bad, result, dict(hidden=128), lambda _: 'binding')
