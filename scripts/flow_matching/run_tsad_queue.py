"""Resume frozen full-data TSAD jobs, preserving failures and other GPU queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.phase_gate import check_queue
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    handle.seek(0)
    handle.write(b'1')
    handle.flush()
    handle.seek(0)
    try:
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def complete_result(root, job):
    path = root / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if record.get('status') != 'completed' or record.get('experiment_sha256') != fingerprint:
        raise ValueError('Completed result belongs to another experiment')
    for name, key in [('scores.npz', 'scores_sha256'), ('model.pt', 'checkpoint_sha256')]:
        if sha(path.parent / name) != record[key]:
            raise ValueError('Completed artifact changed')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    base = ROOT / queue['state_root']
    state_path = base / (queue['lane'] + '_status.json')
    lane_lock = exclusive_lock(base / (queue['lane'] + '.lock'))
    if lane_lock is None:
        raise RuntimeError('This TSAD lane already has a live lock owner')
    gpu_lock = None
    if queue['lane'] == 'gpu':
        while True:
            pending = {path: len(check_queue(ROOT / path)) for path in queue['gpu_prerequisite_queues']}
            active = [p.info['pid'] for p in psutil.process_iter(['pid', 'cmdline'])
                      if any('run_queue.py' in a for a in p.info['cmdline'] or [])]
            write_json(state_path, {'status': 'waiting_existing_imputation', 'pid': os.getpid(),
                                     'pending_prerequisites': pending, 'live_prior_queue_pids': active})
            if not any(pending.values()) and not active:
                gpu_lock = exclusive_lock(ROOT / 'experiments/runs/flow_matching_campaign/queue.lock')
                if gpu_lock is not None:
                    break
            time.sleep(30)
    records = []
    for job in queue['jobs']:
        if complete_result(ROOT, job):
            records.append({'id': job['id'], 'status': 'completed'})
            continue
        output = ROOT / job['output_directory']
        if output.exists():
            records.append({'id': job['id'], 'status': 'partial_artifacts_preserved_pending_recovery'})
            continue
        logs = base / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        command = [queue['python'], '-m', 'scripts.flow_matching.run_tsad_experiment', '--queue', str(args.queue), '--job-id', job['id']]
        environment = {key.upper(): value for key, value in os.environ.items()}
        environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(queue['evaluation']['cpu_threads']), MKL_NUM_THREADS=str(queue['evaluation']['cpu_threads']))
        with (logs / (job['id'] + '.stdout.log')).open('a', encoding='utf-8') as stdout, (logs / (job['id'] + '.stderr.log')).open('a', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            while child.poll() is None:
                write_json(state_path, {'status': 'running', 'pid': os.getpid(), 'active_pid': child.pid, 'active_job': job['id'],
                                        'total_jobs': len(queue['jobs']), 'records': records})
                time.sleep(10)
        completed = complete_result(ROOT, job)
        records.append({'id': job['id'], 'status': 'completed' if completed else 'failed', 'returncode': child.returncode})
        write_json(state_path, {'status': 'running', 'pid': os.getpid(), 'total_jobs': len(queue['jobs']), 'records': records})
    status = 'completed' if all(r['status'] == 'completed' for r in records) else 'completed_with_unresolved_jobs'
    write_json(state_path, {'status': status, 'pid': os.getpid(), 'records': records, 'total_jobs': len(queue['jobs'])})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
