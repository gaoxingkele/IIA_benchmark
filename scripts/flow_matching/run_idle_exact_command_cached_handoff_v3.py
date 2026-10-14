"""Cold handoff of four idle full queues; reuse cached Windows telemetry only.

Original entry modules and every argument remain exact. Numerical children,
resource thresholds, queues, checkpoints and raw data are never modified.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child

SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
ENTRIES = {
    'spectral70': ('scripts.flow_matching.run_spectral_resource_order_v4', 70),
    'industrial9': ('scripts.flow_matching.run_light_controller', 9),
    'strict175': ('scripts.flow_matching.run_bounded_strict_cpu_queue_v2', 175),
    'tranad25': ('scripts.flow_matching.run_tranad_reference_float64_queue_v1', 25),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify(config, name, root=ROOT):
    if config['original_scope'] != SCOPE or set(config['roles']) != set(ENTRIES):
        raise ValueError('The entire original research scope and all four complete queues are required')
    for pin in config['source_receipts']:
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Frozen handoff source changed: ' + pin['path'])
    role = config['roles'][name]
    entry, count = ENTRIES[name]
    command = role['previous_owner']['command']
    index = command.index('-m')
    if (role['entry_module'] != entry or command[index + 1] != entry
            or command[index + 2:] != role['original_arguments']):
        raise ValueError('Exact original entry and every original argument must survive')
    queue = read(root / role['original_queue'])
    if len(queue['jobs']) != count or role['required_original_slots'] != count:
        raise ValueError('Every original complete job must remain')
    key = 'original_id' if name == 'industrial9' else 'id'
    if len({job[key] for job in queue['jobs']}) != count:
        raise ValueError('Original canonical jobs must remain distinct')
    if name == 'industrial9' and any(job['original_job']['id'] != job['original_id'] for job in queue['jobs']):
        raise ValueError('Industrial recovery must retain exact original job identity')
    return role


def idle_allowed(state, expected, descendants):
    # Two legacy state schemas omit create_time. The live OS PID/time/command
    # is checked separately before and after suspension; no timestamp is invented.
    return (state.get('pid') == expected['pid']
            and ('create_time' not in state or state['create_time'] == expected['create_time'])
            and state.get('active_pid') is None
            and state.get('state') in {'waiting_for_memory', 'waiting_for_heavy_lock'}
            and not state.get('shared_lock_owned')
            and all(allowed_console_child(child['name'], child['exe']) for child in descendants))


def exact_live(process, expected):
    return process.create_time() == expected['create_time'] and process.cmdline() == expected['command']


def descendants_of(process):
    return [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]


def handoff(role, base, poll_seconds=10):
    expected = role['previous_owner']
    while True:
        process = psutil.Process(expected['pid'])
        if not exact_live(process, expected):
            raise RuntimeError('Previous exact live process identity changed; do not stop it')
        state = read(ROOT / role['previous_state_path'])
        children = descendants_of(process)
        if not idle_allowed(state, expected, children):
            write(base / 'status.json', dict(state='waiting_for_verified_idle_owner',
                pid=os.getpid(), previous_phase=state.get('state'),
                previous_active_pid=state.get('active_pid')))
            time.sleep(poll_seconds)
            continue
        process.suspend()
        stopped = False
        try:
            if not exact_live(process, expected):
                raise RuntimeError('Suspended owner identity changed')
            state = read(ROOT / role['previous_state_path'])
            children = descendants_of(process)
            if not idle_allowed(state, expected, children):
                continue
            exclusive(base / 'previous_status_at_verified_handoff.json', state)
            exclusive(base / 'handoff_identity.json', dict(previous_owner=expected,
                descendants=children, legacy_state_records_create_time='create_time' in state,
                exact_live_os_pid_time_and_command_checked_twice=True,
                no_numerical_child_present=True, shared_lock_owned=False))
            process.terminate()
            process.wait(timeout=15)
            stopped = True
            exclusive(base / 'handoff.json', dict(previous_owner=expected,
                original_native_child_stopped=False, waiting_controller_only_stopped=True,
                successor_pid=os.getpid(), captured_utc=datetime.now(timezone.utc).isoformat()))
            return
        finally:
            if not stopped and process.is_running():
                process.resume()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', required=True, choices=ENTRIES)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    role = verify(config, args.role)
    install_cached_loader(lambda snapshot: None)
    if args.probe_only:
        importlib.import_module(role['entry_module'])
        from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
        print(json.dumps(dict(role=args.role, complete_original_slots=role['required_original_slots'],
            original_arguments=role['original_arguments'], snapshot=resource_snapshot(),
            private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            original_numerical_child_executed=False)))
        return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base = ROOT / config['handoff_state_root'] / args.role
    base.mkdir(parents=True, exist_ok=True)
    owner = resource_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Exact cached successor already owned')
    try:
        handoff(role, base)
        verify(config, args.role)
        exclusive(base / 'successor_entry.json', dict(entry_module=role['entry_module'],
            original_arguments=role['original_arguments'], configuration_sha256=digest(ROOT / args.config),
            original_scope=SCOPE, required_original_slots=role['required_original_slots'],
            original_numerical_settings_and_guards_unchanged=True))
        sys.argv = [role['entry_module'], *role['original_arguments']]
        runpy.run_module(role['entry_module'], run_name='__main__', alter_sys=True)
    finally:
        owner.close()


if __name__ == '__main__':
    main()
