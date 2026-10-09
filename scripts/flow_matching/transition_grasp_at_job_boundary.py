"""Hand off a guarded native scheduler only after its active model has exited."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

from scripts.flow_matching.grasp_protocol_runner import ROOT, complete, verify, sha, write_json
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def boundary_observation(parent, state, expected, complete_check):
    """A terminal complete child plus no new model is required; all races defer."""
    command = parent.cmdline()
    if (parent.pid != expected['pid'] or parent.create_time() != expected['create_time']
            or expected['module'] not in command or '--queue' not in command):
        return None
    actual = Path(command[command.index('--queue') + 1])
    actual = actual if actual.is_absolute() else ROOT / actual
    if actual.resolve() != Path(expected['queue_path']).resolve():
        return None
    job = state.get('active_job')
    if not job or job not in expected['job_ids']:
        return None
    if state.get('active_pid') and psutil.pid_exists(state['active_pid']):
        return None
    if any(not console_helper(child) for child in parent.children(recursive=True)):
        return None
    if not complete_check(job):
        return None
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    for source in config['source_receipts']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen handoff source changed')
    original_path, successor_path = ROOT / config['original_queue'], ROOT / config['successor_queue']
    if sha(original_path) != config['original_queue_sha256'] or sha(successor_path) != config['successor_queue_sha256']:
        raise ValueError('Frozen native queue changed')
    old_queue = json.loads(original_path.read_text(encoding='utf-8'))
    new_queue = json.loads(successor_path.read_text(encoding='utf-8'))
    verify(old_queue); verify(new_queue)
    if old_queue['jobs'] != new_queue['jobs'] or old_queue['settings'] != new_queue['settings']:
        raise ValueError('Handoff changed full experimental jobs or budgets')
    jobs = {j['id']: j for j in old_queue['jobs']}
    expected = {**config['expected_controller'], 'job_ids': set(jobs)}
    base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    lock = exclusive_lock(base / 'worker.lock')
    if lock is None:
        raise RuntimeError('Native boundary handoff watcher already live')
    receipt = base / 'transition.json'
    if receipt.exists():
        raise FileExistsError('Boundary handoff already recorded')
    state_path = ROOT / old_queue['state_root'] / 'status.json'
    parent = psutil.Process(expected['pid'])
    command = parent.cmdline()
    if (parent.create_time() != expected['create_time'] or expected['module'] not in command
            or '--queue' not in command
            or Path(command[command.index('--queue') + 1]).resolve() != original_path.resolve()):
        raise ValueError('Original guarded scheduler identity differs')
    while True:
        state = json.loads(state_path.read_text(encoding='utf-8'))
        child_pid = state.get('active_pid')
        observed = {'pid': __import__('os').getpid(), 'state': 'waiting_model_completion',
                    'original_controller_pid': parent.pid, 'original_active_pid': child_pid,
                    'original_active_job': state.get('active_job'), 'config_sha256': sha(path)}
        write_json(base / 'status.json', observed)
        if child_pid and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            command = child.cmdline()
            if (child.ppid() != parent.pid or 'scripts.flow_matching.grasp_protocol_cuda_runtime' not in command
                    or state.get('active_job') not in command):
                raise ValueError('Observed child does not match original full model')
            observed.update(child_create_time=child.create_time(), child_command=command)
            write_json(base / 'status.json', observed)
            # Windows process wait preserves the parent's original guard while
            # the model runs. A timeout leaves this same child running.
            try:
                child.wait(timeout=30)
            except psutil.TimeoutExpired:
                continue
        candidate = boundary_observation(parent, state, expected, lambda job: complete(jobs[job]))
        if candidate is None:
            time.sleep(.1)
            continue
        # Suspend only after confirmed model exit. If the original dispatched
        # another model in the meantime, resume immediately and wait again.
        parent.suspend()
        stopped = False
        try:
            again = json.loads(state_path.read_text(encoding='utf-8'))
            confirmed = boundary_observation(parent, again, expected, lambda job: complete(jobs[job]))
            if confirmed != candidate:
                continue
            saved = {'original_controller': expected, 'last_complete_job': confirmed, 'last_state': again,
                     'original_command': parent.cmdline(), 'original_controller_verified_childless': True,
                     'model_processes_stopped': False, 'original_guard_active_until_model_exit': True,
                     'full_jobs_and_budgets_unchanged': True, 'config_sha256': sha(path)}
            saved['original_controller']['job_ids'] = sorted(saved['original_controller']['job_ids'])
            write_json(base / 'before_handoff.json', saved)
            parent.terminate(); parent.wait(timeout=10)
            stopped = True
        finally:
            if not stopped and parent.is_running():
                parent.resume()
        if stopped:
            break
    command = [sys.executable, '-u', '-m', 'scripts.flow_matching.run_grasp_fair_queue_v4', '--queue', str(successor_path)]
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
    with (base / 'successor.stdout.log').open('x', encoding='utf-8') as out, (base / 'successor.stderr.log').open('x', encoding='utf-8') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
    write_json(receipt, {**saved, 'new_pid': child.pid, 'new_command': command, 'handoff_complete': True})
    write_json(base / 'status.json', {**observed, 'state': 'completed_boundary_handoff', 'new_pid': child.pid})


if __name__ == '__main__':
    main()
