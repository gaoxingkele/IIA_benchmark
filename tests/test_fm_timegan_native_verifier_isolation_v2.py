"""Actual full output verification and immutable complete replay fidelity."""
import ast
import builtins
import copy
import inspect
import json

import pytest

from scripts.flow_matching import timegan_native_result_verifier_v2 as isolated
from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v3 as dispatch
from scripts.flow_matching import run_timegan_full45_completion_replay_v5 as replay

ROOT = isolated.common.ROOT


def test_verifier_is_the_exact_complete_original_function():
    tree = ast.parse((ROOT / 'scripts/flow_matching/run_timegan_native_queue_v1.py').read_text(encoding='utf-8'))
    expected = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_result')
    assert ast.dump(isolated.verifier_ast().body[0]) == ast.dump(expected)


def test_native_worker_changes_one_import_only():
    expected = ast.parse(inspect.getsource(isolated.observer.worker))
    actual = isolated.observer_worker_ast()
    for node in ast.walk(actual):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.timegan_native_result_verifier_v2':
            node.module = 'scripts.flow_matching.run_timegan_native_queue_v1'
    assert ast.dump(actual) == ast.dump(expected)


def test_actual_completed_output_verifies_without_host_monitor_import(monkeypatch):
    native_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'psutil' or name.startswith('psutil.'):
            raise AssertionError('Native numerical environment must not import host process monitoring')
        return native_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    queue = isolated.common.read(ROOT / 'configs/experiments/fm_timegan_native_original_queue.v1.json')
    job = next(j for j in queue['jobs'] if j['id'] == 'timegan_native__gru__energy__seed1104')
    result = isolated.verify_result(queue, job)
    assert result['optimizer_updates'] == dict(E0=50000, GS=50000, E=100000, G=100000, D=50000)
    assert len(result['checkpoints']) == 23
    assert len(result['evaluators']) == 20
    assert result['independent_full_checkpoint_replay_completed'] is False


@pytest.mark.parametrize('tamper', ['artifact', 'budget', 'evaluators', 'checkpoints'])
def test_exact_original_failure_gates_remain(tamper):
    namespace = isolated.verifier_namespace()
    queue = isolated.common.read(ROOT / 'configs/experiments/fm_timegan_native_original_queue.v1.json')
    job = next(j for j in queue['jobs'] if j['id'] == 'timegan_native__gru__energy__seed1104')
    result = copy.deepcopy(isolated.common.read(ROOT / job['output_directory'] / 'result.json'))
    if tamper == 'artifact':
        result['artifacts'][0]['sha256'] = '0' * 64
    elif tamper == 'budget':
        result['optimizer_updates']['G'] = 99999
    elif tamper == 'evaluators':
        result['evaluators'].pop()
    else:
        result['checkpoints'].pop()
    original_read = namespace['read']
    result_path = ROOT / job['output_directory'] / 'result.json'
    namespace['read'] = lambda path: result if path == result_path else original_read(path)
    with pytest.raises(ValueError):
        namespace['verify_result'](queue, job)


@pytest.mark.parametrize('tamper', ['guard', 'threads', 'other_controller'])
def test_dispatch_cannot_change_science_or_other_live_recovery(tamper):
    config = isolated.common.read(ROOT / replay.DISPATCH)
    if tamper == 'guard':
        config['roles']['original']['settings']['minimum_commit_headroom_bytes'] = 1
    elif tamper == 'threads':
        config['cpu_threads'] = 1
    else:
        config['roles']['energy']['state_root'] = 'wrong'
    with pytest.raises(ValueError):
        dispatch.verify_runtime(config)


@pytest.mark.parametrize('tamper', ['budget', 'tolerance', 'route', 'receipt'])
def test_all45_replay_contract_remains_strict(tamper):
    config = isolated.common.read(ROOT / 'configs/runtime/fm_timegan_full45_completion_replay.v6.json')
    if tamper == 'budget':
        config['protocol_iterations_each'] = 1
    elif tamper == 'tolerance':
        config['equality_gate'] = 'allclose'
    elif tamper == 'route':
        config['routes'][-1]['execution_job']['seed'] = 42
    else:
        config['routes'][-1]['native_completion_receipts'].pop()
    with pytest.raises(ValueError):
        replay.verify(config)


def test_replay_worker_is_unchanged_and_verifier_binding_is_native_safe():
    prior = replay.original.runtime_namespace()
    actual = replay.runtime_namespace()
    assert actual['worker'].__code__ is prior['worker'].__code__
    assert actual['worker'].__globals__['completed_native_result'] is replay.completed_native_result
    assert any(isinstance(value, str) and
               "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v5'" in value
               for value in actual['main'].__code__.co_consts)
    config = isolated.common.read(ROOT / 'configs/runtime/fm_timegan_full45_completion_replay.v6.json')
    _, routes = replay.verify(config)
    assert len(routes) == 45
    assert config['required_checkpoints_each'] * len(routes) == 1035
    assert config['required_evaluators_each'] * len(routes) == 900


def test_failed_exit_stays_failed_and_independent_receipt_has_explicit_provenance():
    folder = ROOT / 'projects/flow_matching_research/results/2026-10-13/timegan_energy1104_actual_completion_v1'
    evidence = isolated.common.read(folder / 'completion_evidence.json')
    for item in evidence['artifacts']:
        path = folder / item['archive_name']
        assert path.read_bytes() == ROOT.joinpath(item['source_path']).read_bytes()
        assert isolated.common.sha(path) == item['sha256']
    failed = isolated.common.read(folder / 'preserved_worker_exit1_receipt.json')
    verified = isolated.common.read(folder / 'independent_completion_verification.json')
    assert failed['exit_code'] == 1 and failed['full_native_result_verified'] is False
    assert verified['original_training_worker_exit_code'] == 1
    assert verified['full_native_result_verified'] is True
    assert verified['scientific_training_repeated'] is False
    assert verified['strict_TAB_result'] is False
    assert verified['independent_full_checkpoint_replay_completed'] is False
    assert verified['all_original_experiments_complete'] is False
