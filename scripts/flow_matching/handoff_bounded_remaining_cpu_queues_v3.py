"""Handoff only exact idle CPU adapters; preserve completed and failed attempts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from scripts.flow_matching.run_bounded_remaining_cpu_queues_v3 import ROOT, SPECS, bind, digest
from scripts.flow_matching.handoff_bounded_strict_cpu_v2 import observation
from scripts.flow_matching.run_tsad_queue import complete_result


def verify_idle(registration, name, state, observed, terminal):
    expected = registration['handoffs'][name]
    old = expected['predecessor']
    if (observed['pid'] != old['pid'] or observed['create_time'] != old['create_time']
            or observed['command'] != old['command'] or state['pid'] != old['pid']
            or state['create_time'] != old['create_time']
            or state['runtime_sha256'] != registration['parent_resource_registration_sha256']
            or state['queue_sha256'] != registration['controllers'][name]['queue_sha256']
            or state['state'] not in ('waiting_for_memory', 'waiting_for_heavy_lock', 'waiting_resource_slot')
            or state.get('active_pid') is not None or state.get('shared_lock_owned') is not False
            or state['requested_job'] != expected['requested_job']
            or state['requested_command'] != expected['requested_command']
            or any(p['name'].lower() != 'conhost.exe' for p in observed['children'])):
        raise ValueError('Exact idle owned adapter and no actual model child required')
    if state.get('active_job') is None:
        if terminal is not None:
            raise ValueError('No stale job may acquire an unrelated terminal proof')
    elif (terminal is None or terminal['job_id'] != state['active_job']
            or terminal['job_id'] == state['requested_job']
            or terminal['queue_sha256'] != state['queue_sha256']
            or terminal['create_time'] != state['active_create_time']
            or terminal['command'] != state['actual_command']
            or terminal['exit_code'] != state['exit_code']
            or terminal['resource_guard_aborted'] != state['resource_guard_aborted']
            or not terminal['full_registered_budget_retained']):
        raise ValueError('Stale previous job requires its exact successful or preserved-failed terminal proof')
    return dict(actual_process=observed, original_status=state, previous_terminal=terminal, verified_idle=True)


def check_pending(root, name, job):
    output = root / job['output_directory']
    if output.exists():
        entries = list(output.rglob('*'))
        if name != 'pi_cpu' or any(not p.is_file() or p.name not in ('stdout.log', 'stderr.log')
                                    or p.stat().st_size != 0 for p in entries):
            raise ValueError('Preserve any existing pending model artifacts or nonempty logs')
    return dict(output_directory=job['output_directory'], exists=output.exists(),
                only_original_empty_pi_logs=name == 'pi_cpu' and output.exists())


def verify_current(registration, name):
    expected = registration['handoffs'][name]
    state = json.loads((ROOT / expected['predecessor_status']).read_text(encoding='utf-8'))
    terminal_path = expected.get('previous_terminal_receipt')
    terminal = json.loads((ROOT / terminal_path).read_text(encoding='utf-8')) if terminal_path else None
    process = psutil.Process(expected['predecessor']['pid'])
    record = verify_idle(registration, name, state, observation(process), terminal)
    queue = json.loads((ROOT / registration['controllers'][name]['queue']).read_text(encoding='utf-8'))
    if terminal:
        previous = next(j for j in queue['jobs'] if j['id'] == terminal['job_id'])
        if terminal['exit_code'] == 0:
            if name != 'entropic_cpu' or not complete_result(ROOT, previous):
                raise ValueError('Previous successful full model must independently verify')
        else:
            failure = json.loads((ROOT / previous['output_directory'] / 'failure.json').read_text(encoding='utf-8'))
            if (name != 'pi_cpu' or failure.get('status') != 'failed_preserved'
                    or failure.get('id') != terminal['job_id'] or failure.get('exit_code') != terminal['exit_code']
                    or not failure.get('logs_preserved')):
                raise ValueError('Original failed attempt must remain explicitly unresolved and preserved')
    job = next(j for j in queue['jobs'] if j['id'] == state['requested_job'])
    record['pending_artifact_check'] = check_pending(ROOT, name, job)
    for other in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        command = other.info['cmdline'] or []
        if '--job-id' in command and command[command.index('--job-id') + 1] == job['id']:
            raise ValueError('An actual registered model worker already exists, including detached children')
    return process, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', choices=SPECS, required=True)
    args = parser.parse_args()
    registration = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    bind(registration, args.controller)
    name = args.controller
    process, record = verify_current(registration, name)
    report = ROOT / registration['report_directory'] / name
    report.mkdir(parents=True, exist_ok=True)
    receipt_path = report / 'handoff_receipt.json'
    snapshot = report / 'preserved_idle_status.json'
    if receipt_path.exists() or snapshot.exists():
        raise FileExistsError('Preserve original per-controller handoff evidence')
    shutil.copyfile(ROOT / registration['handoffs'][name]['predecessor_status'], snapshot)
    process, record = verify_current(registration, name)
    process.terminate()
    code = process.wait(timeout=10)
    command = [registration['python'], '-X', 'utf8', '-u', '-m',
               'scripts.flow_matching.run_bounded_remaining_cpu_queues_v3', '--runtime-config', args.runtime_config,
               '--controller', name]
    with (report / 'controller.stdout.log').open('xb') as out, (report / 'controller.stderr.log').open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                     PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        handle = psutil.Process(child.pid)
        launched = dict(pid=handle.pid, create_time=handle.create_time(), command=handle.cmdline())
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), runtime=args.runtime_config,
        runtime_sha256=digest(ROOT / args.runtime_config), retired_verified_idle_controller=record,
        old_terminal_exit_code=code, preserved_idle_status_sha256=digest(snapshot), launched_controller=launched,
        controller=name, registered_original_jobs=SPECS[name][0], original_cpu_threads=SPECS[name][1],
        resource_lock=registration['resource_lock'], original_model_workers_stopped=0,
        additional_training_seed_slots=0, new_performance_rows=0, all_original_experiments_complete=False)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(controller=name, full_original_jobs=SPECS[name][0], launched_controller=launched,
                          model_workers_stopped=0, additional_bounded_training_concurrency=0)))


if __name__ == '__main__':
    main()
