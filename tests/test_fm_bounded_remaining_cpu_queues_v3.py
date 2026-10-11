import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.run_bounded_remaining_cpu_queues_v3 import ROOT, SPECS, bind
from scripts.flow_matching.handoff_bounded_remaining_cpu_queues_v3 import check_pending, verify_idle


def idle_case(name='entropic_cpu', failed=False):
    predecessor = dict(pid=10, create_time=1, command=['python', 'original-owned-adapter'])
    registration = dict(parent_resource_registration_sha256='parent', controllers={name: dict(queue_sha256='queue')},
        handoffs={name: dict(predecessor=predecessor, requested_job='next',
                            requested_command=['python', '--job-id', 'next'])})
    state = dict(pid=10, create_time=1, runtime_sha256='parent', queue_sha256='queue', state='waiting_for_memory',
        active_pid=None, shared_lock_owned=False, requested_job='next',
        requested_command=registration['handoffs'][name]['requested_command'], active_job='previous',
        active_create_time=2, actual_command=['python', '--job-id', 'previous'], exit_code=int(failed),
        resource_guard_aborted=failed)
    observed = dict(predecessor, children=[dict(name='conhost.exe')])
    terminal = dict(job_id='previous', create_time=2, command=state['actual_command'], queue_sha256='queue',
                    exit_code=int(failed), resource_guard_aborted=failed, full_registered_budget_retained=True)
    return registration, name, state, observed, terminal


@pytest.mark.parametrize('failed', [False, True])
def test_exact_previous_success_or_failed_attempt_never_becomes_a_live_child(failed):
    args = idle_case('pi_cpu' if failed else 'entropic_cpu', failed)
    saved = copy.deepcopy(args)
    assert verify_idle(*args)['verified_idle']
    assert args == saved


@pytest.mark.parametrize('change', ['pid_reused', 'native_child', 'active_worker', 'shared_lock', 'wrong_command',
                                   'wrong_queue', 'wrong_job', 'wrong_previous_exit', 'wrong_previous_handle',
                                   'running', 'missing_terminal'])
def test_handoff_never_retires_unowned_or_running_process(change):
    registration, name, state, observed, terminal = idle_case()
    if change == 'pid_reused': observed['create_time'] = 9
    elif change == 'native_child': observed['children'].append(dict(name='python.exe'))
    elif change == 'active_worker': state['active_pid'] = 20
    elif change == 'shared_lock': state['shared_lock_owned'] = True
    elif change == 'wrong_command': observed['command'] = ['unowned']
    elif change == 'wrong_queue': state['queue_sha256'] = 'wrong'
    elif change == 'wrong_job': state['requested_job'] = 'other'
    elif change == 'wrong_previous_exit': terminal['exit_code'] = 1
    elif change == 'wrong_previous_handle': terminal['create_time'] = 3
    elif change == 'running': state['state'] = 'running'
    elif change == 'missing_terminal': terminal = None
    with pytest.raises(ValueError): verify_idle(registration, name, state, observed, terminal)


def test_moment_idle_without_stale_job_needs_no_previous_terminal():
    registration, name, state, observed, terminal = idle_case('moment_cpu')
    state['active_job'] = None
    assert verify_idle(registration, name, state, observed, None)['verified_idle']
    with pytest.raises(ValueError): verify_idle(registration, name, state, observed, terminal)


def test_only_empty_original_pi_logs_can_be_reopened(tmp_path):
    job = dict(output_directory='pending')
    output = tmp_path / 'pending'
    output.mkdir()
    for name in ['stdout.log', 'stderr.log']: (output / name).touch()
    assert check_pending(tmp_path, 'pi_cpu', job)['only_original_empty_pi_logs']
    with pytest.raises(ValueError): check_pending(tmp_path, 'moment_cpu', job)
    (output / 'stdout.log').write_text('preserved actual attempt', encoding='utf-8')
    with pytest.raises(ValueError): check_pending(tmp_path, 'pi_cpu', job)


def test_checkpoint_cannot_be_overwritten_even_with_empty_pi_logs(tmp_path):
    job = dict(output_directory='pending')
    output = tmp_path / 'pending'
    output.mkdir()
    (output / 'checkpoints').mkdir()
    with pytest.raises(ValueError): check_pending(tmp_path, 'pi_cpu', job)


def registration():
    return json.loads((ROOT / 'configs/runtime/fm_bounded_remaining_cpu_queues.v3.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', list(SPECS))
def test_entire_original_queue_and_all_command_threads_preserved(name):
    function = bind(registration(), name)
    from scripts.flow_matching import run_resource_gated_light_controller_v1 as original
    assert function.__code__ is original.main.__code__
    assert function.__globals__['guarded_factory'].__code__ is original.guarded_factory.__code__


@pytest.mark.parametrize('change', ['lock', 'settings', 'binding', 'parent_hash', 'subset'])
def test_no_new_concurrency_weakened_guard_or_scientific_rebinding(change):
    value = registration()
    if change == 'lock': value['resource_lock'] += '.different'
    elif change == 'settings': value['settings']['minimum_available_memory_bytes'] -= 1
    elif change == 'binding': value['controllers']['pi_cpu']['registered_jobs'] -= 1
    elif change == 'parent_hash': value['parent_resource_registration_sha256'] = 'changed'
    elif change == 'subset': del value['controllers']['moment_cpu']
    with pytest.raises(ValueError): bind(value, 'entropic_cpu')


def test_same_real_serial_resource_lock_is_exclusive(tmp_path):
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    strict = json.loads((ROOT / registration()['existing_bounded_strict_registration']).read_text(encoding='utf-8'))
    assert registration()['resource_lock'] == strict['resource_lock']
    first = exclusive_lock(tmp_path / 'serial.lock')
    assert first is not None
    try:
        assert exclusive_lock(tmp_path / 'serial.lock') is None
    finally:
        first.close()
    next_owner = exclusive_lock(tmp_path / 'serial.lock')
    assert next_owner is not None
    next_owner.close()
