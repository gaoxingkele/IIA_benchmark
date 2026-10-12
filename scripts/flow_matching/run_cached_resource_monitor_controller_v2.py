"""Cached Windows telemetry for two verified idle original experiment controllers.

Native dispatch, queues, validators, budgets and original resource policies are
unchanged. Legacy status without creation time requires an independent pinned
historical launch observation, exact current PID/ctime/command and queue hashes.
Only runtime status receives explicit creation time metadata for future audits.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import allowed_console_child


def historical_identity(config):
    evidence = config.get('historical_identity')
    if evidence is None:
        return False
    if sha(ROOT / evidence['path']) != evidence['sha256']:
        raise ValueError('Independent historical owner observation changed')
    document = read(ROOT / evidence['path'])
    matches = [r for r in document['controllers'] if r['name'] == evidence['controller_name']]
    if len(matches) != 1:
        raise ValueError('One independent historical controller identity required')
    old = config['handoff']['previous_owner']; record = matches[0]
    if (record['pid'] != old['pid'] or record['create_time'] != old['create_time']
            or record['command_token'] != config['entry_module']
            or record['state']['queue_sha256'] != config['expected_state_identity']['queue_sha256']
            or record['state']['runtime_sha256'] != config['expected_state_identity']['runtime_sha256']):
        raise ValueError('Independent exact historical PID/ctime/module/queue/runtime identity required')
    return True


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen cached monitor binding changed: ' + pin['path'])
    if config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273):
        raise ValueError('Complete original research scope required')
    module = importlib.import_module(config['entry_module'])
    runtime = read(ROOT / config['original_runtime_configuration'])
    if config['entry_module'] == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
        if config['original_arguments'] != ['--runtime-config', config['original_runtime_configuration'], '--controller', 'original']:
            raise ValueError('All original CFM jobs must remain selected')
        selected, queue = module.binding(runtime, 'original')
        settings, lock = queue['resource_policy'], runtime['family_lock']
        if len(queue['jobs']) != 285:
            raise ValueError('All285 original CFM tasks required')
        status_path = queue['state_root'] + '/status.json'
        queue_path = selected['queue']
        if not historical_identity(config):
            raise ValueError('Legacy CFM state requires independently pinned historical identity')
    elif config['entry_module'] == 'scripts.flow_matching.run_saits_guarded_original_queue_v1':
        if config['original_arguments'] != ['--runtime-config', config['original_runtime_configuration']]:
            raise ValueError('Entire original SAITS/BRITS queue required')
        queue = module.verify_binding(runtime)
        settings, lock = runtime['settings'], runtime['resource_lock']
        if len(queue['jobs']) != 210 or config.get('historical_identity') is not None:
            raise ValueError('All210 original imputation tasks and explicit status creation time required')
        status_path = runtime['state_root'] + '/status.json'
        queue_path = runtime['queue']
    else:
        raise ValueError('Unregistered host controller must not be migrated')
    identity = dict(queue_sha256=sha(ROOT / queue_path), runtime_sha256=sha(ROOT / config['original_runtime_configuration']))
    if (settings != config['settings'] or lock != config['resource_lock']
            or config['handoff']['previous_state_path'] != status_path or config['expected_state_identity'] != identity):
        raise ValueError('Exact full queue, native policy, family lock and status path required')
    return module, queue


def safe_phase(config, state, descendants):
    expected = config['handoff']['previous_owner']
    if (state.get('pid') != expected['pid'] or state.get('active_pid') is not None
            or state.get('state') not in ('waiting_for_memory', 'waiting_for_heavy_lock', 'between_jobs', 'between_full_jobs')
            or any(not allowed_console_child(p['name'], p['exe']) for p in descendants)
            or any(state.get(k) != v for k, v in config['expected_state_identity'].items())):
        return False
    if state.get('create_time') is None:
        return historical_identity(config)
    return state['create_time'] == expected['create_time']


def safe_handoff(config, base, update):
    import psutil
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive
    expected = config['handoff']['previous_owner']
    path = ROOT / config['handoff']['previous_state_path']

    def identity(process):
        return process.create_time() == expected['create_time'] and process.cmdline() == expected['command']

    def descendants(process):
        return [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]

    while True:
        process = psutil.Process(expected['pid'])
        if not identity(process):
            raise RuntimeError('Previous PID reused or command changed; never stop it')
        state = read(path); children = descendants(process)
        if not safe_phase(config, state, children):
            update(dict(state='waiting_verified_idle_handoff', previous_state=state.get('state'),
                        previous_active_pid=state.get('active_pid'), previous_descendants=children))
            time.sleep(10)
            continue
        process.suspend(); stopped = False
        try:
            if not identity(process):
                raise RuntimeError('Suspended owner identity changed')
            state = read(path); children = descendants(process)
            if not safe_phase(config, state, children):
                continue
            exclusive(base / 'previous_status_at_verified_handoff.json', state)
            exclusive(base / 'handoff_identity.json', dict(previous_owner=expected, descendants=children,
                independent_historical_identity_used=state.get('create_time') is None,
                no_native_child_present=True))
            process.terminate(); process.wait(timeout=15); stopped = True
            write(path, dict(state, state='stopped_for_verified_cached_snapshot_handoff', active_pid=None,
                successor_pid=os.getpid(), successor_configuration=config['runtime_configuration'],
                original_native_child_stopped=False))
            exclusive(base / 'handoff.json', dict(previous_owner=expected, only_idle_controller_stopped=True,
                no_native_child_present=True, original_native_training_interrupted=False))
            return
        finally:
            if not stopped and process.is_running():
                process.resume()


def install_status_identity(module, config, created):
    original_write = module.write
    target = (ROOT / config['handoff']['previous_state_path']).resolve()

    def status_write(path, data):
        if Path(path).resolve() == target and isinstance(data, dict) and data.get('pid') == os.getpid():
            data = dict(data, create_time=created, resource_monitor_configuration=config['runtime_configuration'])
        return original_write(path, data)

    module.write = status_write
    return status_write


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.runtime_config)
    if config['runtime_configuration'] != args.runtime_config:
        raise ValueError('Exact immutable monitor configuration required')
    module, queue = verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, full_original_jobs=len(queue['jobs']), original_guards_unchanged=True,
            training_executed=False, native_dispatch_source_unchanged=True)))
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base = ROOT / config['monitor_state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = resource_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Cached host monitor already owned')
    created = psutil.Process().create_time()
    state = dict(pid=os.getpid(), create_time=created, active_pid=None, state='starting',
        configuration_sha256=sha(ROOT / args.runtime_config), all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)

    recorded = [False]

    def first_snapshot(result):
        if not recorded[0]:
            write(base / 'actual_first_cached_snapshot.json', dict(snapshot=result, pid=os.getpid(), create_time=created,
                original_settings=config['settings'], original_queue_jobs=len(queue['jobs'])))
            recorded[0] = True

    try:
        update({}); safe_handoff(config, base, update)
        install_cached_loader(first_snapshot)
        install_status_identity(module, config, created)
        update(dict(state='running_original_controller_with_cached_resource_telemetry'))
        sys.argv = [config['entry_module']] + config['original_arguments']
        module.main()
        update(dict(state='original_controller_returned'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
