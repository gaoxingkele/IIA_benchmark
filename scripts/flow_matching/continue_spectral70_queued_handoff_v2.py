"""Queue for the real GPU lock, then safely resume all70 frozen reexecutions.

Only exact idle metadata parents may be retired after actual lock acquisition.
No running numerical child is stopped or changed. Completed/failed attempts
and every original numerical program, command, budget and guard are retained.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

from scripts.flow_matching import continue_spectral70_fair_conditions_v1 as previous
from scripts.flow_matching.win32_queued_byte_lock_v1 import QueuedByteLock
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic

ROOT, read, sha = previous.ROOT, previous.read, previous.sha
MODULE = 'scripts.flow_matching.continue_spectral70_queued_handoff_v2'
CONFIG = 'configs/runtime/fm_spectral70_queued_handoff.v2.json'
TEST = 'tests/test_fm_spectral70_queued_handoff_v2.py'
BASE = 'experiments/runs/fm_spectral70_queued_handoff_v2'
EXPECTED_OBSERVER = dict(pid=39164, create_time=1792142098.9482229, command=[
    'C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe',
    '-B', '-X', 'utf8', '-u', '-m', previous.MODULE])


def registration():
    prior = read(ROOT/previous.CONFIG)
    c, parent, cases = previous.verify(prior)
    pins = {x['path'] for x in prior['source_receipts']}
    pins.update([previous.CONFIG, MODULE.replace('.', '/')+'.py', TEST,
                 'scripts/flow_matching/win32_queued_byte_lock_v1.py'])
    return dict(schema_version=2, state_root=BASE, previous_configuration=previous.CONFIG,
        previous_configuration_sha256=sha(ROOT/previous.CONFIG),
        original_configuration=previous.original.CONFIG,
        original_configuration_sha256=sha(ROOT/previous.original.CONFIG),
        previous_owner=copy.deepcopy(prior['previous_owner']), previous_observer=copy.deepcopy(EXPECTED_OBSERVER),
        required_conditions=70, required_cohorts=14, original_scope=c['original_scope'],
        settings=c['settings'], spectral_flow_minimum_commit_headroom_bytes=c['spectral_flow_minimum_commit_headroom_bytes'],
        resource_lock=c['resource_lock'], lock_offset=0, lock_bytes=1,
        pending_lock_poll_seconds=.25, inter_condition_yield_seconds=35,
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(pins)],
        documentation=[
            'https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex',
            'https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex'],
        numerical_worker_or_configuration_changed=False, all_original_experiments_complete=False,
        boundary='Queued Win32 exclusive byte0,length1 on same original GPU lock; never bypass ownership. '
          'Acquire actual lock before exact old parent and idle v1 observer identity checked twice. '
          'No numeric descendants; terminal child guard; every attempt represented as verified or preserved failure. '
          'Rehash full successful reports and immutable artifacts before retirement; retain all outcomes without retry. '
          'Only metadata parents retired; original numeric module/config/command/solver/budgets/resources retained. '
          'Reuse existing safe continuation metadata AST with only v2 metadata configuration binding and35second yield. '
          'Keep all70/14 and full45/681/273/702 scope; no new training credit from handoff.')


def verify(config):
    if config != registration():
        raise ValueError('All70 original numerical bindings, exact identities and queued lock required')
    return previous.verify(read(ROOT/previous.CONFIG))


def exact_process(expected):
    import psutil
    process = psutil.Process(expected['pid'])
    if process.create_time() != expected['create_time'] or process.cmdline() != expected['command']:
        raise ValueError('Exact metadata process identity changed; preserve all work')
    return process


def descendants(process):
    return [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]


def observer_allowed(state, expected, children):
    return (state.get('pid') == expected['pid'] and state.get('create_time') == expected['create_time']
        and state.get('active_pid') is None and state.get('required_conditions') == 70
        and state.get('required_cohorts') == 14
        and state.get('state') == 'waiting_actual_metric_lock_release_no_process_stopped'
        and all(previous.fair.allowed_console_child(p['name'], p['exe']) for p in children))


def continuation_program(c, old, reports, oldlock, base):
    # Clone function globals privately. The live v1 module and its configuration
    # stay unchanged, including every original numeric AST and subprocess call.
    ns = dict(previous.__dict__, CONFIG=CONFIG, MODULE=MODULE)
    ns['initial_state'] = FunctionType(previous.initial_state.__code__, ns)
    compile_program = FunctionType(previous.controller_program.__code__, ns)
    return compile_program(c, old, reports, oldlock, base)


def handoff(config, c, parent, cases, api, base, update):
    status = ROOT/c['state_root']/'status.json'
    observer_base = ROOT/read(ROOT/previous.CONFIG)['state_root']
    paused, retired = [], set()
    kernel = None
    oldlock = observerlock = None
    try:
        while True:
            if (observer_base/'handoff.json').exists():
                update(state='v1_handoff_already_completed_no_replacement_or_numeric_started')
                return None
            owner = exact_process(config['previous_owner'])
            observer = exact_process(config['previous_observer'])
            if kernel is None:
                kernel = QueuedByteLock(ROOT/config['resource_lock'])
                update(state='original_GPU_lock_request_submitted_to_Windows', kernel_lock_owned=kernel.acquired)
            if not kernel.poll():
                update(state='waiting_pending_Windows_GPU_lock_no_process_stopped', kernel_lock_owned=False)
                time.sleep(config['pending_lock_poll_seconds'])
                continue
            update(state='actual_original_GPU_lock_owned_checking_idle_metadata', kernel_lock_owned=True)
            old = read(status)
            prior_state = read(observer_base/'status.json')
            attempts = [p.name for p in (ROOT/c['state_root']/'attempts').iterdir() if p.is_dir()]
            if (not previous.boundary_allowed(old, config['previous_owner'], descendants(owner),
                    previous.terminal_child(c, old), attempts)
                    or not observer_allowed(prior_state, config['previous_observer'], descendants(observer))):
                # Acquisition alone is insufficient. Release and wait, preserving
                # active children, unknown attempts or any other ambiguous state.
                kernel.close(); kernel = None
                update(state='boundary_not_proven_original_processes_preserved', kernel_lock_owned=False)
                time.sleep(config['pending_lock_poll_seconds'])
                continue
            owner.suspend(); paused.append(owner)
            observer.suspend(); paused.append(observer)
            exact_process(config['previous_owner']); exact_process(config['previous_observer'])
            old, prior_state = read(status), read(observer_base/'status.json')
            attempts = [p.name for p in (ROOT/c['state_root']/'attempts').iterdir() if p.is_dir()]
            old_desc, observer_desc = descendants(owner), descendants(observer)
            if (not previous.boundary_allowed(old, config['previous_owner'], old_desc,
                    previous.terminal_child(c, old), attempts)
                    or not observer_allowed(prior_state, config['previous_observer'], observer_desc)):
                raise ValueError('Second idle metadata boundary proof failed; do not retire processes')
            # Validate all old scientific outcomes BEFORE retiring either parent.
            reports, bindings = previous.restored_reports(c, parent, cases, old)
            for name, path in [('previous_status_original_bytes.json', status),
                               ('previous_observer_status_original_bytes.json', observer_base/'status.json')]:
                with (base/name).open('xb') as f:
                    f.write(path.read_bytes())
            previous.fair.exclusive(base/'handoff_identity.json', dict(previous_owner=config['previous_owner'],
                previous_observer=config['previous_observer'], owner_descendants=old_desc,
                observer_descendants=observer_desc, actual_original_GPU_byte_lock_owned=True,
                exact_identities_checked_twice=True, terminal_child_guard_verified=True,
                all_attempts_have_preserved_terminal_outcomes=True,
                numerical_child_stopped=False, numerical_configuration_changed=False))
            previous.fair.exclusive(base/'restored_verified_report_bindings.json',
                dict(records=bindings, failed_preserved=old['failed_preserved'], new_numerical_completions=0))
            # These are now proven idle metadata processes, with the real shared
            # GPU lock held. No process tree termination is used.
            observer.terminate(); observer.wait(timeout=15); retired.add(observer.pid)
            owner.terminate(); owner.wait(timeout=15); retired.add(owner.pid)
            oldlock = api['exclusive_lock'](ROOT/c['state_root']/'controller.lock')
            observerlock = api['exclusive_lock'](observer_base/'controller.lock')
            if oldlock is None or observerlock is None:
                raise RuntimeError('Original metadata owner lock unavailable; refuse duplicate continuation')
            previous.fair.exclusive(base/'handoff.json', dict(previous_owner=config['previous_owner'],
                previous_observer=config['previous_observer'], successor_pid=os.getpid(),
                captured_utc=datetime.now(timezone.utc).isoformat(),
                only_two_proven_idle_metadata_parents_retired=True, numerical_child_stopped=False,
                verified_conditions_preserved=len(reports), failures_preserved=len(old['failed_preserved'])))
            return old, reports, oldlock, observerlock
    except BaseException:
        if oldlock is not None: oldlock.close()
        if observerlock is not None: observerlock.close()
        raise
    finally:
        for process in reversed(paused):
            if process.pid not in retired and process.is_running():
                process.resume()
        if kernel is not None:
            kernel.close()


def run(config, c, parent, cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api = install_facades()
    base = ROOT/config['state_root']; base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('Queued continuation already owned')
    if (base/'status.json').exists(): lane.close(); raise FileExistsError('Preserve every previous metadata attempt')
    process = psutil.Process()
    state = dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(), active_pid=None,
        required_conditions=70, required_cohorts=14, all_original_experiments_complete=False,
        scheduler_configuration=CONFIG, scheduler_configuration_sha256=sha(ROOT/CONFIG))
    def update(value=None, **kw):
        state.update(value or {}, **kw, captured_utc=datetime.now(timezone.utc).isoformat())
        atomic(base/'status.json', state)
    locks = None
    try:
        update(state='waiting_for_original_full_numeric_condition_no_process_stopped')
        locks = handoff(config, c, parent, cases, api, base, update)
        if locks is None: return
        old, reports, oldlock, observerlock = locks
        update(state='yielding_shared_GPU_lock_after_queued_safe_handoff', kernel_lock_owned=False)
        time.sleep(config['inter_condition_yield_seconds'])
        program, tree = continuation_program(c, old, reports, oldlock, base)
        program(c, parent, cases)
    finally:
        if locks is not None:
            locks[2].close(); locks[3].close()
        lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        previous.fair.exclusive(ROOT/CONFIG, registration()); return
    config = read(ROOT/CONFIG); c, parent, cases = verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, conditions=len(cases), cohorts=14,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            actual_GPU_lock_requested=False, numerical_worker_executed=False)))
    else:
        run(config, c, parent, cases)


if __name__ == '__main__':
    main()
