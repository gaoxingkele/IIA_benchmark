"""Replace only an observed scheduler after its full model worker exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from scripts.flow_matching.run_light_controller import ROOT, load_controller, digest
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def completed_job(root, job, kind, namespace):
    if kind != 'pi':
        return namespace['complete_result'](root, job)
    path = root / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    receipt = json.loads(path.read_text(encoding='utf-8'))
    expected = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if receipt['experiment_sha256'] != expected:
        raise ValueError('Completed Pi job identity changed')
    namespace['verify_artifacts'](receipt, root)
    return True


def boundary_observation(parent, state, expected, completed, pid_exists=psutil.pid_exists,
                         helper_check=console_helper):
    if (parent.pid != expected['pid'] or parent.create_time() != expected['create_time']
            or parent.cmdline() != expected['command'] or state.get('pid') != parent.pid):
        return None
    if state.get('active_pid') and pid_exists(state['active_pid']):
        return None
    if any(not helper_check(child) for child in parent.children(recursive=True)):
        return None
    job = state.get('active_job')
    if job:
        return job if job in expected['job_ids'] and completed(job) else None
    return 'verified_idle_boundary'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--controller', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    runtime = json.loads(config_path.read_text(encoding='utf-8'))
    _, namespace, binding = load_controller(ROOT, runtime, args.controller)
    queue = json.loads((ROOT / binding['queue_path']).read_text(encoding='utf-8'))
    jobs = {job['id']: job for job in queue['jobs']}
    expected = dict(binding['predecessor'], job_ids=set(jobs))
    base = ROOT / binding['transition_root']; base.mkdir(parents=True, exist_ok=True)
    lock = namespace['exclusive_lock'](base / 'watcher.lock')
    if lock is None:
        raise RuntimeError('A boundary watcher already owns this lane')
    if (base / 'transition.json').exists():
        raise FileExistsError('Preserve successful boundary handoff')
    write = namespace['write_json']
    checker = lambda job: completed_job(ROOT, jobs[job], binding['completion_kind'], namespace)
    state_path = ROOT / binding['status_path']
    parent = psutil.Process(expected['pid'])
    if parent.create_time() != expected['create_time'] or parent.cmdline() != expected['command']:
        raise ValueError('Predecessor PID/time/command no longer matches')
    while True:
        state = json.loads(state_path.read_text(encoding='utf-8'))
        observation = {'pid': os.getpid(), 'state': 'waiting_model_completion',
            'predecessor_pid': parent.pid, 'predecessor_active_pid': state.get('active_pid'),
            'predecessor_active_job': state.get('active_job'), 'runtime_sha256': digest(config_path)}
        write(base / 'status.json', observation)
        try:
            candidate = boundary_observation(parent, state, expected, checker)
        except psutil.NoSuchProcess:
            # A terminal/missing scheduler requires fresh orphan/queue validation.
            # A watcher never treats an observation timeout as a stopped model.
            write(base / 'status.json', dict(observation, state='predecessor_missing_requires_revalidation'))
            return
        if candidate is None:
            active = state.get('active_pid')
            if active and psutil.pid_exists(active):
                try:
                    child = psutil.Process(active)
                    if child.ppid() == parent.pid:
                        child.wait(timeout=1)
                    else:
                        time.sleep(1)
                except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                    pass
            else:
                time.sleep(.1)
            continue
        # Suspend only a scheduler with no live model children. Recheck every
        # identity and completion condition while suspended to close dispatch races.
        parent.suspend(); stopped = False
        try:
            again = json.loads(state_path.read_text(encoding='utf-8'))
            if boundary_observation(parent, again, expected, checker) != candidate:
                continue
            saved = {'predecessor': binding['predecessor'], 'predecessor_state': again,
                'last_complete_job_or_idle_boundary': candidate,
                'predecessor_private_bytes': parent.memory_info().private,
                'runtime_sha256': digest(config_path), 'queue_sha256': binding['queue_sha256'],
                'full_jobs_seeds_budgets_workers_unchanged': True, 'model_processes_stopped': False}
            write(base / 'before_handoff.json', saved)
            parent.terminate(); parent.wait(timeout=10); stopped = True
        finally:
            if not stopped and parent.is_running():
                parent.resume()
        break
    command = [sys.executable, '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.run_light_controller',
        '--runtime-config', args.config, '--controller', args.controller]
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
             if sys.platform == 'win32' else 0)
    with (base / 'successor.stdout.log').open('x', encoding='utf-8') as out, (base / 'successor.stderr.log').open('x', encoding='utf-8') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'), creationflags=flags)
    write(base / 'transition.json', dict(saved, successor_pid=child.pid,
        successor_create_time=psutil.Process(child.pid).create_time(), successor_command=command))
    write(base / 'status.json', dict(observation, state='completed_boundary_handoff', successor_pid=child.pid))


if __name__ == '__main__':
    main()
