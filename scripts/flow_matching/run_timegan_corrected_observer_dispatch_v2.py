"""Safely hand off only waiting controllers; retain all45 original native jobs.

Never suspend/stop an original native child. Original controller handoff waits
for a recorded between-job phase without an active PID or Python descendants.
Stage recovery handoff also requires its output directory to remain absent.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_predictive_observer_v2 import ROOT, read, sha, verify_runtime, worker
from scripts.flow_matching.timegan_native_common_v1 import write


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def allowed_console_child(name, executable):
    return (name.lower() == 'conhost.exe' and Path(executable).as_posix().lower()
        == 'c:/windows/system32/conhost.exe')


def handoff_phase_allowed(role, state, output_exists, descendants):
    if state.get('active_pid') is not None or any(not allowed_console_child(p['name'], p['exe']) for p in descendants):
        return False
    if role != 'original' and output_exists:
        return False
    allowed = {'starting', 'waiting_for_memory', 'waiting_for_resource_lock', 'waiting_for_family_lock',
        'waiting_for_lock', 'between_full_jobs', 'between_jobs'}
    return state.get('state') in allowed


def live_identity(process, expected):
    return process.create_time() == expected['create_time'] and process.cmdline() == expected['command']


def safe_handoff(role, route, base, update, poll_seconds):
    import psutil
    expected = route['previous_owner']
    while True:
        try:
            process = psutil.Process(expected['pid'])
        except psutil.NoSuchProcess:
            state = read(ROOT / route['previous_state_path'])
            if state.get('state') not in ('completed', 'completed_with_unresolved_jobs', 'failed_attempt_preserved'):
                raise RuntimeError('Original owner disappeared without a verified terminal state; inspect preserved attempt')
            if state.get('active_pid') is not None:
                raise RuntimeError('Terminal state still names an unverified native child')
            exclusive(base / 'handoff.json', dict(previous_owner=expected, previous_terminal_state=state,
                original_native_child_stopped=False, original_owner_already_terminal=True))
            return
        if not live_identity(process, expected):
            raise RuntimeError('Previous PID reused or native controller command changed; never terminate it')
        state = read(ROOT / route['previous_state_path'])
        if state.get('pid') != expected['pid'] or state.get('create_time') != expected['create_time']:
            raise RuntimeError('Previous exact owner/state identity changed')
        descendants = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]
        output_exists = bool(route.get('output_directory') and (ROOT / route['output_directory']).exists())
        if not handoff_phase_allowed(role, state, output_exists, descendants):
            update(dict(state='waiting_for_verified_native_between_job_handoff',
                actual_verified_previous_owner=expected, previous_phase=state.get('state'),
                previous_active_pid=state.get('active_pid'), previous_descendants=descendants))
            time.sleep(poll_seconds)
            continue
        process.suspend()
        stopped = False
        try:
            if not live_identity(process, expected):
                raise RuntimeError('Suspended owner identity changed')
            state = read(ROOT / route['previous_state_path'])
            descendants = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]
            output_exists = bool(route.get('output_directory') and (ROOT / route['output_directory']).exists())
            if not handoff_phase_allowed(role, state, output_exists, descendants):
                continue
            exclusive(base / 'previous_status_at_verified_handoff.json', state)
            exclusive(base / 'handoff_identity.json', dict(previous_owner=expected, descendants=descendants,
                output_directory_exists=output_exists, phase_verified=state['state'],
                no_native_child_present=True, only_controller_will_be_stopped=True))
            process.terminate()
            process.wait(timeout=15)
            stopped = True
            previous = dict(state, state='stopped_for_verified_handoff', active_pid=None,
                successor_configuration=route['runtime_configuration'], successor_role=role,
                successor_pid=os.getpid(), original_native_child_stopped=False,
                captured_utc=datetime.now(timezone.utc).isoformat())
            write(ROOT / route['previous_state_path'], previous)
            exclusive(base / 'handoff.json', dict(previous_owner=expected, no_native_child_present=True,
                original_native_child_stopped=False, waiting_controller_only_stopped=True))
            return
        finally:
            if not stopped and process.is_running():
                process.resume()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', required=True, choices=['original', 'stock', 'energy'])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--worker-probe-only', action='store_true')
    parser.add_argument('--job-id')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if config['runtime_configuration'] != args.config:
        raise ValueError('Exact immutable runtime configuration required')
    queue = verify_runtime(config)
    if args.worker or args.worker_probe_only:
        report = worker(config, args.role, args.job_id, args.worker_probe_only)
        if report is not None:
            print(json.dumps(report))
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.run_timegan_native_queue_v1 import verify_result
    from scripts.flow_matching.timegan_stage_resume_v2 import verify_recovery_result
    api = resource_api()
    route = config['roles'][args.role]
    base = ROOT / route['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Corrected observer controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', role=args.role,
        active_pid=None, configuration_sha256=sha(ROOT / args.config), jobs={},
        full_original_scope_jobs=45, independent_training_seed_slots_added=0,
        all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(base / 'status.json', state)

    try:
        update({})
        safe_handoff(args.role, route, base, update, config['poll_seconds'])
        jobs = queue['jobs'] if args.role == 'original' else [route['canonical_job']]
        for job in jobs:
            output = ROOT / (job['output_directory'] if args.role == 'original' else route['output_directory'])
            if (output / 'result.json').exists():
                if args.role == 'original':
                    verify_result(queue, job)
                else:
                    verify_recovery_result(read(ROOT / route['recovery_configuration']))
                state['jobs'][job['id']] = 'completed_preexisting'
                update({})
                continue
            if output.exists():
                state['jobs'][job['id']] = 'failed_or_partial_preserved_requires_exact_recovery'
                update({})
                continue
            update(dict(active_job=job['id']))
            with api['resource_slot'](ROOT / config['resource_lock'], route['settings'], update):
                bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");"
                    "sys.argv=['corrected_timegan_observer','--config'," + repr(args.config) + ","
                    "'--role'," + repr(args.role) + ",'--worker','--job-id'," + repr(job['id']) + "];"
                    "runpy.run_module('scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2',run_name='__main__')")
                command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / (job['id'] + '.stdout.log')).open('xb') as out, (logs / (job['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                            OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_original_timegan_with_corrected_observation', active_pid=child.pid,
                        active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                    code, usage = capped_wait(child, dict(queue, settings=route['settings']), api, update)
            passed = False
            if code == 0 and (output / 'result.json').exists():
                if args.role == 'original':
                    verify_result(queue, job)
                else:
                    verify_recovery_result(read(ROOT / route['recovery_configuration']))
                passed = True
            write(base / 'receipts' / (job['id'] + '.json'), dict(exit_code=code,
                full_native_result_verified=passed, predictive_target_observation_corrected=True, **usage))
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved_requires_exact_recovery'
            update(dict(state='between_full_jobs', active_pid=None, active_job=None))
            time.sleep(config['poll_seconds'])
        update(dict(state='completed' if all(v in ('completed', 'completed_preexisting') for v in state['jobs'].values())
            else 'completed_with_unresolved_jobs', full_original_scope_jobs=45))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
