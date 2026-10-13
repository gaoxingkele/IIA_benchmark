"""Full numerical replay gates and immutable45-slot route extension."""
import copy
from unittest.mock import patch

import pytest

from scripts.flow_matching import timegan_saved_joint_full_replay_v1 as replay

CONFIG = 'configs/runtime/fm_timegan_sine_saved_joint15000_full_replay.v1.json'


def test_only_pending_sine_result_route_changes():
    config = replay.read(replay.ROOT / CONFIG)
    queue, routes = replay.verify(config)
    previous = replay.read(replay.ROOT / replay.PREVIOUS)
    assert len(queue['jobs']) == len(routes) == 45
    changed = [(a, b) for a, b in zip(previous['routes'], routes) if a != b]
    assert len(changed) == 1
    assert changed[0][0]['canonical_job'] == changed[0][1]['canonical_job']
    assert changed[0][1]['canonical_job']['id'] == replay.CANONICAL
    assert changed[0][1]['kind'] == 'saved_joint_continuation'


def test_every_original_numerical_worker_and_gate_code_identical():
    namespace = replay.native_namespace()
    for name in ['worker', 'complete_metadata', 'bitwise_array', 'bitwise_score',
                 'restore_full_checkpoint', 'numpy_state_matches', 'check_report', 'protocol_contract']:
        assert namespace[name].__code__ is getattr(replay.BASE, name).__code__
        assert namespace[name].__globals__ is namespace
    assert namespace['completed_native_result'] is replay.completed_native_result
    assert namespace['verify'] is replay.verify


@pytest.mark.parametrize('field,value', [('required_checkpoints_each', 22), ('required_evaluators_each', 19),
    ('protocol_iterations_each', 69999), ('equality_gate', 'approximate')])
def test_no_reduced_full_native_gate(field, value):
    config = replay.read(replay.ROOT / CONFIG); config[field] = value
    with pytest.raises(ValueError, match='scope/budget/guard'):
        replay.verify(config)


def test_other44_routes_cannot_be_rewritten():
    config = copy.deepcopy(replay.read(replay.ROOT / CONFIG))
    route = next(r for r in config['routes'] if r['canonical_job']['id'] != replay.CANONICAL)
    route['result_path'] = 'different/result.json'
    with pytest.raises(ValueError, match='All45 routes'):
        replay.verify(config)


def test_new_route_requires_full_continuation_verifier_and_proof_digest():
    config = replay.read(replay.ROOT / CONFIG)
    route = next(r for r in config['routes'] if r['canonical_job']['id'] == replay.CANONICAL)
    recovery = replay.read(replay.ROOT / replay.RECOVERY)
    with patch.object(replay, 'read', side_effect=[recovery, {'saved_joint_native_proof_sha256': 'a' * 64}]), \
         patch.object(replay.continuation, 'verify_result', side_effect=ValueError('not complete')) as verify:
        with pytest.raises(ValueError, match='not complete'):
            replay.completed_native_result(route, {})
    verify.assert_called_once_with(recovery, 'a' * 64)


def test_actual_existing_certificates_adopted_without_training_or_failed_case_credit():
    config = replay.read(replay.ROOT / CONFIG)
    union = replay.collect_full45_evidence(config)
    assert union['required_original_models'] == 45
    assert 'timegan_native__gru__stock__seed1103' in union['verified_original_ids']
    assert 'timegan_native__gru__energy__seed1104' in union['verified_original_ids']
    assert 'timegan_native__gru__stock__seed1104' not in union['verified_original_ids']
    assert replay.CANONICAL not in union['verified_original_ids']
    assert union['pending_original_models'] == 45-len(union['verified_original_ids'])
    assert union['all_original_experiments_complete'] is False
