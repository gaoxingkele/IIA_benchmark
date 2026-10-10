"""Replace only the idle GiFlow scheduler after its existing full model exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def candidate(parent, state, expected, pid_exists=psutil.pid_exists, helper_check=console_helper):
    if (parent.pid != expected['pid'] or parent.create_time() != expected['create_time']
            or parent.cmdline() != expected['command'] or state.get('pid') != parent.pid):
        return False
    if state.get('active_pid') and pid_exists(state['active_pid']):
        return False
    return not any(not helper_check(child) for child in parent.children(recursive=True))


def compare_full_jobs(old, new):
    if old['settings'] != new['settings'] or old['resource_lock'] != new['resource_lock']:
        raise ValueError('Original resources or guards changed')
    fields = ('id', 'dataset', 'track', 'recipe', 'seed', 'arguments', 'author_source')
    a = [{k:j[k] for k in fields} for j in old['jobs']]
    b = [{k:j[k] for k in fields} for j in new['jobs']]
    if a != b:
        raise ValueError('Original full experiment scope, algorithms or budgets changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args(); path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned handoff implementation changed')
    old_path, new_path = ROOT / config['original_queue'], ROOT / config['successor_queue']
    if sha(old_path) != config['original_queue_sha256'] or sha(new_path) != config['successor_queue_sha256']:
        raise ValueError('Pinned full queues changed')
    old = json.loads(old_path.read_text(encoding='utf-8')); new = json.loads(new_path.read_text(encoding='utf-8'))
    verify(old); verify(new); compare_full_jobs(old, new)
    from scripts.flow_matching.run_giflow_serial_native_queue_v4 import memory_preflight_ready
    if not memory_preflight_ready(new):
        raise ValueError('Full-batch GPU memory proof missing')
    base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    owner = exclusive_lock(base / 'watcher.lock')
    if owner is None or (base / 'transition.json').exists():
        raise RuntimeError('Preserve existing GiFlow boundary handoff')
    expected = config['expected_controller']; parent = psutil.Process(expected['pid'])
    if parent.create_time() != expected['create_time'] or parent.cmdline() != expected['command']:
        raise ValueError('Live predecessor identity differs')
    state_path = ROOT / old['state_root'] / 'status.json'
    try:
        while True:
            state = json.loads(state_path.read_text(encoding='utf-8'))
            observed = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                state='waiting_existing_full_model_exit', predecessor_pid=parent.pid,
                predecessor_active_pid=state.get('active_pid'), predecessor_active_job=state.get('active_job'),
                config_sha256=sha(path))
            write_json(base / 'status.json', observed)
            if not candidate(parent, state, expected):
                active = state.get('active_pid')
                if active and psutil.pid_exists(active):
                    child = psutil.Process(active)
                    if child.ppid() != parent.pid or 'scripts.flow_matching.run_giflow_native_job' not in child.cmdline():
                        raise ValueError('Existing full model identity differs')
                    try:
                        child.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        continue
                else:
                    time.sleep(.1)
                continue
            parent.suspend(); stopped = False
            try:
                again = json.loads(state_path.read_text(encoding='utf-8'))
                if not candidate(parent, again, expected):
                    continue
                saved = dict(predecessor=expected, last_state=again, full_scope_and_budgets_unchanged=True,
                    predecessor_verified_childless=True, model_processes_stopped=False,
                    predecessor_guard_active_until_model_exit=True, config_sha256=sha(path))
                write_json(base / 'before_handoff.json', saved)
                parent.terminate(); parent.wait(timeout=10); stopped = True
            finally:
                if not stopped and parent.is_running():
                    parent.resume()
            if stopped:
                break
        command = [sys.executable, '-X', 'utf8', '-u', '-m',
            'scripts.flow_matching.run_giflow_serial_native_queue_v4', '--queue', str(new_path)]
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        with (base / 'successor.stdout.log').open('xb') as out, (base / 'successor.stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
        write_json(base / 'transition.json', dict(saved, successor_pid=child.pid,
            successor_create_time=psutil.Process(child.pid).create_time(), successor_command=command, handoff_complete=True))
        write_json(base / 'status.json', dict(observed, state='completed_boundary_handoff', successor_pid=child.pid))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
