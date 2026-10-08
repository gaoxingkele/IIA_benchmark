"""Run one complete zero-shot queue on CPU; preserve existing partial artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock, complete_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    base = ROOT / queue['state_root']
    lock = exclusive_lock(base / 'moment_pretrained.lock')
    if lock is None:
        raise RuntimeError('MOMENT worker already owns lock')
    state_path = base / 'moment_pretrained_status.json'
    sources = {r['path']: r['sha256'] for j in queue['jobs'] for r in j['frozen_files']}
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError('MOMENT frozen source/data differs: ' + name)
    records = []
    for job in queue['jobs']:
        if complete_result(ROOT, job):
            records.append({'id': job['id'], 'status': 'completed'})
            continue
        if (ROOT / job['output_directory']).exists():
            records.append({'id': job['id'], 'status': 'partial_preserved'})
            continue
        logs = base / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        env = {k.upper(): v for k, v in os.environ.items()}
        env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS=str(queue['evaluation']['cpu_threads']), MKL_NUM_THREADS=str(queue['evaluation']['cpu_threads']))
        with (logs / (job['id'] + '.stdout.log')).open('a', encoding='utf-8') as out, (logs / (job['id'] + '.stderr.log')).open('a', encoding='utf-8') as err:
            child = subprocess.Popen([queue['python'], '-u', str(ROOT / 'scripts/flow_matching/run_moment_pretrained_job.py'),
                                      '--queue', str(args.queue), '--job-id', job['id']], cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            while child.poll() is None:
                write_json(state_path, {'status': 'running', 'pid': os.getpid(), 'active_pid': child.pid, 'active_job': job['id'], 'records': records,
                                        'registered_jobs': len(queue['jobs']), 'frozen_files_verified': len(sources)})
                time.sleep(10)
        done = complete_result(ROOT, job)
        records.append({'id': job['id'], 'status': 'completed' if done else 'failed_preserved', 'returncode': child.returncode})
    write_json(state_path, {'status': 'completed' if all(r['status'] == 'completed' for r in records) else 'completed_with_unresolved_jobs',
                           'pid': os.getpid(), 'records': records, 'registered_jobs': len(queue['jobs'])})


if __name__ == '__main__':
    main()
