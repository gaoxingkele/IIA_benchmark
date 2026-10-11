"""Strengthen future GiFlow host-memory admission at a verified full-job boundary.

The exact original full jobs, native worker, outputs and CUDA batch proof stay
unchanged. No active model or its original resource guard is interrupted.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching import run_giflow_serial_native_queue_v4 as native
from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


RESOURCE_CHANGES = {'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'}
QUEUE_CHANGES = {'id', 'state_root', 'settings', 'parent_admission_queue',
                 'parent_admission_queue_sha256', 'commit_reserve_policy'}


def compare_full_scope(old, new):
    if old['jobs'] != new['jobs'] or len(old['jobs']) != 140:
        raise ValueError('Every original full algorithm, model, seed, output and job must remain exact')
    if set(old['settings']) != set(new['settings']):
        raise ValueError('No resource or scientific setting may disappear')
    for key, value in old['settings'].items():
        if key in RESOURCE_CHANGES:
            if new['settings'][key] < value:
                raise ValueError('Never relax a native memory safeguard')
        elif new['settings'][key] != value:
            raise ValueError('Original scientific and GPU settings must stay exact')
    if ({k:v for k,v in old.items() if k not in QUEUE_CHANGES}
            != {k:v for k,v in new.items() if k not in QUEUE_CHANGES}):
        raise ValueError('Only registered host commit admission and controller state location may change')
    settings = new['settings']
    policy = new['commit_reserve_policy']
    if (settings['emergency_commit_headroom_bytes'] < policy['shared_CPU_emergency_floor_bytes']
            or settings['minimum_commit_headroom_bytes'] < policy['registered_full_worker_private_bytes']
                + settings['emergency_commit_headroom_bytes'] + policy['concurrent_growth_margin_bytes']
            or new['controller_yield_seconds'] != old['controller_yield_seconds']):
        raise ValueError('Full native measured private allocation, shared reserve and margin must fit')


def eligible(parent, state, expected, last_job, complete_check, pid_exists=psutil.pid_exists,
             helper_check=console_helper):
    if (parent.pid != expected['pid'] or parent.create_time() != expected['create_time']
            or parent.cmdline() != expected['command'] or state.get('pid') != parent.pid
            or state.get('queue_sha256') != expected['queue_sha256']):
        return False
    if state.get('active_pid') and pid_exists(state['active_pid']):
        return False
    if state.get('active_job') or any(not helper_check(p) for p in parent.children(recursive=True)):
        return False
    return bool(last_job and state.get('jobs', {}).get(last_job) == 'completed' and complete_check(last_job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    for entry in config['source_receipts']:
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen commit-reserve boundary evidence differs: ' + entry['path'])
    old_path, new_path = ROOT / config['original_queue'], ROOT / config['successor_queue']
    if (sha(old_path) != config['original_queue_sha256']
            or sha(new_path) != config['successor_queue_sha256']):
        raise ValueError('Frozen full native queues changed')
    old = json.loads(old_path.read_text(encoding='utf-8'))
    new = json.loads(new_path.read_text(encoding='utf-8'))
    verify(old)
    verify(new)
    compare_full_scope(old, new)
    if not native.memory_preflight_ready(new):
        raise ValueError('Original actual full-batch CUDA memory proof is required')
    if args.probe_only:
        print(json.dumps(dict(all140_original_jobs_exact=True, original_full_batch_memory_proof_passed=True,
            new_minimum_commit_headroom_bytes=new['settings']['minimum_commit_headroom_bytes'],
            original_full_worker='scripts.flow_matching.run_giflow_native_job_v2',
            model_processes_stopped=False, additional_training_seed_slots=0)))
        return
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = exclusive_lock(base / 'watcher.lock')
    if owner is None or (base / 'transition.json').exists():
        raise RuntimeError('Preserve an existing commit-reserve handoff')
    expected = config['expected_controller']
    parent = psutil.Process(expected['pid'])
    if parent.create_time() != expected['create_time'] or parent.cmdline() != expected['command']:
        raise ValueError('Original live controller identity differs')
    jobs = {j['id']:j for j in old['jobs']}
    last_job = config['observed_full_active_job']
    state_path = ROOT / old['state_root'] / 'status.json'
    try:
        while True:
            state = json.loads(state_path.read_text(encoding='utf-8'))
            active = state.get('active_job')
            if active:
                if active not in jobs:
                    raise ValueError('Original controller has an unregistered job')
                last_job = active
            observed = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                state='waiting_verified_full_native_job_exit', predecessor_pid=parent.pid,
                predecessor_active_pid=state.get('active_pid'), predecessor_active_job=active,
                tracked_original_full_job=last_job, configuration_sha256=sha(path),
                model_processes_stopped=False)
            write_json(base / 'status.json', observed)
            check = lambda identity: native.complete(jobs[identity])
            if not eligible(parent, state, expected, last_job, check):
                time.sleep(.25)
                continue
            parent.suspend()
            stopped = False
            try:
                again = json.loads(state_path.read_text(encoding='utf-8'))
                if not eligible(parent, again, expected, last_job, check):
                    continue
                usage_path = ROOT / jobs[last_job]['output_directory'] / 'resource_usage.json'
                usage = json.loads(usage_path.read_text(encoding='utf-8'))
                if usage['resource_guard_aborted']:
                    raise ValueError('A memory-aborted attempt is not a complete boundary')
                saved = dict(predecessor=expected, last_state=again, last_complete_job=last_job,
                    last_complete_resource_receipt=dict(path=usage_path.relative_to(ROOT).as_posix(),
                        sha256=sha(usage_path), resource_guard_aborted=False),
                    full140_jobs_models_data_outputs_and_budgets_exact=True,
                    predecessor_guard_active_until_complete_model_exit=True,
                    predecessor_verified_childless=True, model_processes_stopped=False,
                    original_and_successor_queue_sha256=[sha(old_path),sha(new_path)],
                    configuration_sha256=sha(path))
                write_json(base / 'before_handoff.json', saved)
                parent.terminate()
                parent.wait(timeout=10)
                stopped = True
            finally:
                if not stopped and parent.is_running():
                    parent.resume()
            if stopped:
                break
        command = [sys.executable,'-X','utf8','-u','-m',
            'scripts.flow_matching.run_giflow_serial_native_queue_v4','--queue',str(new_path)]
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.DETACHED_PROCESS) if sys.platform == 'win32' else 0
        with (base/'successor.stdout.log').open('xb') as out, (base/'successor.stderr.log').open('xb') as err:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
        write_json(base/'transition.json',dict(saved,successor_pid=child.pid,
            successor_create_time=psutil.Process(child.pid).create_time(),successor_command=command,
            handoff_complete=True))
        write_json(base/'status.json',dict(observed,state='completed_verified_full_job_boundary_handoff',
            successor_pid=child.pid))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
