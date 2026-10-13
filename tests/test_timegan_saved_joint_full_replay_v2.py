"""Full45 certificate integration, host memory bound and unchanged science."""
import ast
import copy
import hashlib
import inspect
from unittest.mock import patch

import pytest

from scripts.flow_matching import timegan_saved_joint_full_replay_v2 as current

CONFIG = 'configs/runtime/fm_timegan_sine_saved_joint15000_full_replay.v2.json'


@pytest.mark.parametrize('length', [0, 1, 65535, 65536, 65537, 220001])
def test_streaming_hash_covers_every_byte_and_boundary(tmp_path, length):
    path = tmp_path / 'data.bin'
    payload = (bytes(range(256)) * (length // 256 + 1))[:length]
    path.write_bytes(payload)
    assert current.bounded_sha(path) == hashlib.sha256(payload).hexdigest()


def test_actual_full45_routes_and_every_original_guard_retained():
    config = current.BASE.read(current.ROOT / CONFIG)
    queue, routes = current.verify(config)
    previous = current.BASE.read(current.ROOT / current.PREVIOUS_SUPPLEMENT)
    assert len(queue['jobs']) == len(routes) == 45
    assert routes == previous['routes']
    assert config['settings'] == previous['settings']


@pytest.mark.parametrize('field,value', [('required_checkpoints_each', 22),
    ('required_evaluators_each', 19), ('protocol_iterations_each', 69999),
    ('equality_gate', 'approximate'), ('host_hash_buffer_bytes', 1048576)])
def test_scope_or_mechanism_cannot_be_relaxed(field, value):
    config = copy.deepcopy(current.BASE.read(current.ROOT / CONFIG))
    config[field] = value
    with pytest.raises(ValueError):
        current.verify(config)


def test_full_numerical_worker_and_all_gate_bytecode_unchanged():
    namespace = current.native_namespace()
    for name in ['worker', 'complete_metadata', 'bitwise_array', 'bitwise_score',
                 'restore_full_checkpoint', 'numpy_state_matches', 'check_report', 'protocol_contract']:
        assert namespace[name].__code__ is getattr(current.BASE, name).__code__
        assert namespace[name].__globals__ is namespace
    assert namespace['verify'] is current.verify


def test_parent_ast_changes_only_native_bootstrap_module():
    actual = current.main_ast()
    for node in ast.walk(actual):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.timegan_saved_joint_full_replay_v2',
                                            'scripts.flow_matching.timegan_saved_joint_full_replay_v1')
    assert ast.dump(actual) == ast.dump(ast.parse(inspect.getsource(current.original.main)))


def test_actual_three_certificates_adopted_and_no_incomplete_sine_credit():
    union = current.collect_full45_evidence(current.BASE.read(current.ROOT / CONFIG))
    assert union['required_original_models'] == 45
    assert union['verified_original_ids'] == ['timegan_native__gru__energy__seed1104',
        'timegan_native__gru__stock__seed1103', 'timegan_native__gru__stock__seed1104']
    assert current.CANONICAL not in union['verified_original_ids']
    assert union['pending_original_models'] == 42
    assert not union['all_original_experiments_complete']


def test_live_exact_prior_owner_prevents_duplicate_dispatch():
    config = current.BASE.read(current.ROOT / CONFIG)
    incident = current.BASE.read(current.ROOT / config['stopped_parent_incident'])
    with patch('psutil.Process') as process:
        process.return_value.create_time.return_value = incident['previous_owner']['create_time']
        process.return_value.cmdline.return_value = incident['previous_owner']['command']
        with pytest.raises(RuntimeError, match='still exists'):
            current.retired_owner_absent(config)


def test_failed_guarded_stock_receipt_never_counts_as_completed():
    config = current.BASE.read(current.ROOT / CONFIG)
    original_read = current.original.read
    stock_config = original_read(current.ROOT / current.STOCK)
    failed_path = current.ROOT / stock_config['state_root'] / 'receipt.json'
    def patched_read(path):
        value = original_read(path)
        if str(path) == str(failed_path):
            value.update(exit_code=1, complete_full_native_replay_passed=False)
        return value
    with patch.object(current.original, 'read', side_effect=patched_read):
        union = current.collect_full45_evidence(config)
    assert 'timegan_native__gru__stock__seed1104' not in union['verified_original_ids']
    assert len(union['verified_original_ids']) == 2
