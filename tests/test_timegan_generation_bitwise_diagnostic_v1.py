import numpy as np
import pytest

from scripts.flow_matching.diagnose_timegan_generation_bitwise_v1 import (
    DESCRIPTION, ObservationComplete, describe_arrays, observed_worker)


def strict(actual, expected, np, description):
    if not np.array_equal(actual, expected):
        raise ValueError('Full finite bitwise equality required: ' + description)
    return 'strict_digest'


def fake_worker(actual, expected, description):
    return bitwise_array(actual, expected, np, description)


def test_single_ULP_is_reported_without_rounding_or_tolerance():
    original = np.array([.5, .25], dtype=np.float32)
    changed = original.copy()
    changed[0] = np.nextafter(changed[0], np.float32(1))
    report = describe_arrays(changed, original, np)
    assert not report['array_equal']
    assert report['unequal_coordinates'] == 1
    assert report['maximum_float32_ULP_distance'] == 1
    assert report['maximum_absolute_difference'] > 0


@pytest.mark.parametrize('different', [False, True])
def test_one_observation_never_becomes_full_replay_completion(different):
    from types import FunctionType
    original = FunctionType(fake_worker.__code__, dict(fake_worker.__globals__, bitwise_array=strict))
    captured = []
    probe = observed_worker(original, lambda a, b, np, evidence: captured.append(evidence))
    assert probe.__code__ is original.__code__
    assert original.__globals__['bitwise_array'] is strict
    with pytest.raises(ObservationComplete):
        probe(np.array([1. if different else 0.], dtype=np.float32), np.array([0.], dtype=np.float32), DESCRIPTION)
    assert len(captured) == 1
    assert captured[0]['original_bitwise_gate_passed'] is (not different)


def test_other_original_checks_still_fail_before_capture():
    from types import FunctionType
    original = FunctionType(fake_worker.__code__, dict(fake_worker.__globals__, bitwise_array=strict))
    captured = []
    probe = observed_worker(original, lambda *args: captured.append(args))
    with pytest.raises(ValueError, match='full author model normalization'):
        probe(np.array([1.]), np.array([0.]), 'full author model normalization')
    assert not captured


def test_nonfinite_or_different_shapes_have_no_numeric_success_claim():
    report = describe_arrays(np.array([np.inf]), np.array([1., 2.]), np)
    assert not report['all_finite'] and not report['same_shape'] and not report['array_equal']
    assert 'maximum_absolute_difference' not in report
