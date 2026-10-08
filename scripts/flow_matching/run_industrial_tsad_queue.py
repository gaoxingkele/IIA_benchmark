"""Execute frozen industrial jobs while preserving prior jobs and GPU ownership."""
from __future__ import annotations

import argparse
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
from scripts.flow_matching.prepare_tsad_execution import write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock, complete_result


def live_prior_gpu_owners(root, paths):
    targets = {str((root/p).resolve()).replace('\\', '/').lower() for p in paths}
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if any('run_queue.py' in a for a in args):
            found.append(process.pid)
        elif any('run_tsad_queue.py' in a for a in args) and '--queue' in args:
            i = args.index('--queue')+1
            if i < len(args) and str((root/args[i]).resolve()).replace('\\', '/').lower() in targets:
                found.append(process.pid)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    base = ROOT/queue['state_root']
    state_path = base/(queue['lane']+'_status.json')
    lane_lock = exclusive_lock(base/(queue['lane']+'.lock'))
    if lane_lock is None:
        raise RuntimeError('Industrial lane already has an active owner')
    gpu_lock = None
    if queue['lane'] == 'gpu':
        while True:
            pending = {p: len(check_queue(ROOT/p)) for p in queue['gpu_prerequisite_queues']}
            prior = live_prior_gpu_owners(ROOT, queue['gpu_prior_tsad_queues'])
            write_json(state_path, {'status': 'waiting_prior_gpu_jobs', 'pid': os.getpid(), 'pending_prerequisites': pending, 'live_prior_gpu_pids': prior})
            if not any(pending.values()) and not prior:
                gpu_lock = exclusive_lock(ROOT/'experiments/runs/flow_matching_campaign/queue.lock')
                if gpu_lock is not None:
                    break
            time.sleep(30)
    records = []
    for job in queue['jobs']:
        if complete_result(ROOT, job):
            records.append({'id': job['id'], 'status': 'completed'})
            continue
        if (ROOT/job['output_directory']).exists():
            records.append({'id': job['id'], 'status': 'partial_artifacts_preserved_pending_recovery'})
            continue
        logs = base/'logs'
        logs.mkdir(parents=True, exist_ok=True)
        command = [queue['python'], '-m', 'scripts.flow_matching.run_industrial_tsad_experiment', '--queue', str(args.queue), '--job-id', job['id']]
        environment = {k.upper(): v for k, v in os.environ.items()}
        environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(queue['evaluation']['cpu_threads']), MKL_NUM_THREADS=str(queue['evaluation']['cpu_threads']))
        if queue['lane'] == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        with (logs/(job['id']+'.stdout.log')).open('a', encoding='utf-8') as stdout, (logs/(job['id']+'.stderr.log')).open('a', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            while child.poll() is None:
                write_json(state_path, {'status': 'running', 'pid': os.getpid(), 'active_pid': child.pid, 'active_job': job['id'], 'total_jobs': len(queue['jobs']), 'records': records})
                time.sleep(10)
        done = complete_result(ROOT, job)
        records.append({'id': job['id'], 'status': 'completed' if done else 'failed', 'returncode': child.returncode})
        write_json(state_path, {'status': 'running', 'pid': os.getpid(), 'records': records, 'total_jobs': len(queue['jobs'])})
    status = 'completed' if all(r['status'] == 'completed' for r in records) else 'completed_with_unresolved_jobs'
    write_json(state_path, {'status': status, 'pid': os.getpid(), 'records': records, 'total_jobs': len(queue['jobs'])})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
