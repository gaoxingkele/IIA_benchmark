"""The metadata fix cannot modify numerical work or admit scientific failures."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import pytest
from scripts.flow_matching import run_moment_TAB100_metadata_recovery_v2 as subject


def test_exactly_one_metadata_predicate_changes_and_no_other_function():
    source = (subject.ROOT/subject.SOURCE).read_text(encoding='utf-8')
    private = subject.corrected_private_module(source)
    old = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    # All numerical workers come from the exact original compiled source.
    for name in ['native_worker', 'review_worker', 'observed_score', 'groups', 'check_source']:
        original = getattr(subject.original, name)
        corrected = getattr(private, name)
        if name != 'check_source':
            assert original.__code__.co_code == corrected.__code__.co_code
            assert original.__code__.co_consts == corrected.__code__.co_consts
        else:
            assert original.__code__.co_code != corrected.__code__.co_code
    assert private.MODULE == subject.MODULE
    assert private._registered_checkpoint_length is subject.registered_checkpoint_length


@pytest.mark.parametrize('replacement', ['True', "audit['configured_sequence_length'] == job['window']"])
def test_rejects_missing_diagnosed_predicate(replacement):
    source = (subject.ROOT/subject.SOURCE).read_text(encoding='utf-8')
    assert subject.OLD_COMPARISON in source
    with pytest.raises(ValueError, match='exactly'):
        subject.corrected_private_module(source.replace(subject.OLD_COMPARISON, replacement))


def test_39_slots_retain_full_original_scope_and_original_resource_policy():
    config = subject.registration()
    runtime, cases = subject.verify(config)
    parent = subject.read(subject.ROOT/subject.ORIGINAL)
    assert len(cases) == 39 and config['required_original_cases'] == 78
    assert all(c['job']['window'] == 100 for c in cases.values())
    assert len([c for c in cases.values() if c['family'] == 'original']) == 36
    assert len([c for c in cases.values() if c['family'] == 'constant_fit']) == 3
    assert runtime['settings'] == parent['settings']
    assert runtime['resource_lock'] == parent['resource_lock']
    assert runtime['mse_tolerance'] == parent['mse_tolerance']
    assert runtime['array_comparison'] == parent['array_comparison']


@pytest.mark.parametrize('change', ['settings', 'selected_ids', 'correction', 'source_receipts'])
def test_changed_limits_scope_correction_or_pins_rejected(change):
    config = subject.registration()
    if change == 'settings': config[change]['minimum_commit_headroom_bytes'] -= 1
    elif change == 'selected_ids': config[change].pop()
    elif change == 'correction': config[change]['external_window'] = 512
    else: config[change][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='Unchanged'):
        subject.verify(config)


def test_checkpoint_length_is_independent_of_external_window():
    parent = subject.read(subject.ROOT/subject.ORIGINAL)
    cases = subject.original.verify(parent)
    for identity in ['fm_moment_pretrained_small__TAB100__psm__seed42', 'fm_moment_pretrained_small__release512__psm__seed42']:
        case = cases[identity]
        assert subject.registered_checkpoint_length(case) == 512
    assert cases['fm_moment_pretrained_small__TAB100__psm__seed42']['job']['window'] == 100


def test_wrong_checkpoint_configuration_hash_rejected():
    case = deepcopy(next(iter(subject.original.verify(subject.read(subject.ROOT/subject.ORIGINAL)).values())))
    case['config']['parameters']['config_sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash changed'):
        subject.registered_checkpoint_length(case)


@pytest.mark.parametrize('change', ['passed', 'exit_code', 'resource_guard_aborted', 'proof_sha256', 'command', 'live'])
def test_scientific_failure_guard_mismatch_or_live_child_cannot_be_retried(tmp_path, monkeypatch, change):
    import psutil
    parent = subject.read(subject.ROOT/subject.ORIGINAL)
    identity = 'fm_moment_pretrained_small__TAB100__psm__seed42'
    case = subject.original.verify(parent)[identity]
    monkeypatch.setattr(subject, 'ROOT', tmp_path)
    target = tmp_path/parent['state_root']/'attempts'/identity/'native'
    target.mkdir(parents=True)
    expected = [str(tmp_path/case['queue']['python']), '-B', '-X', 'utf8', '-u', '-m', subject.original.MODULE,
                '--config', subject.ORIGINAL, '--stage', 'native', '--job-id', identity]
    guard = dict(passed=False, exit_code=1, resource_guard_aborted=False, proof_sha256=None,
                 exact_child_handle=dict(pid=os.getpid(), create_time=psutil.Process().create_time(), command=expected))
    if change == 'passed': guard['passed'] = True
    elif change == 'exit_code': guard['exit_code'] = 0
    elif change == 'resource_guard_aborted': guard[change] = True
    elif change == 'proof_sha256': guard[change] = 'actual-scientific-proof'
    elif change == 'command': guard['exact_child_handle']['command'] = ['another-worker']
    (target/'guard_receipt.json').write_text(json.dumps(guard), encoding='utf-8')
    original_read = subject.read
    monkeypatch.setattr(subject, 'read', lambda p: parent if p == tmp_path/subject.ORIGINAL else original_read(p))
    with pytest.raises(ValueError):
        subject.metadata_failure(case, parent)
