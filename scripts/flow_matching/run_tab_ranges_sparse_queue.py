"""Follow every original range job with isolated exact-affiliation evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.heavy_resources import guarded_wait
from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock
from scripts.flow_matching.tab_sparse_execution import reuse_completed

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    for source in config['source_receipts']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    base = ROOT / config['output_root']
    lock = exclusive_lock(base / 'evaluation.lock')
    if lock is None:
        raise RuntimeError('Sparse range controller already live')
    jobs = [j for p in config['queue_paths'] for j in json.loads((ROOT / p).read_text(encoding='utf-8'))['jobs']]
    state = {'pid': os.getpid(), 'config_sha256': sha(path), 'active_pid': None, 'active_job': None, 'registered_attempts': len(jobs)}
    def update(values):
        state.update(values)
        write_json(base / 'status.json', state)
    complete, failed = {}, {}
    while True:
        waiting = []
        superseded = {j['supersedes_failed_numerics_job']: j['id'] for j in jobs
                      if j.get('supersedes_failed_numerics_job') and j['id'] in complete}
        for job in jobs:
            if job['id'] in complete or job['id'] in failed or job['id'] in superseded:
                continue
            result = ROOT / job['output_directory'] / 'result.json'
            if not result.exists():
                waiting.append(job['id'])
                continue
            cached = reuse_completed(ROOT, result, config)
            if cached is not None:
                complete[job['id']] = {'evaluation_identity': cached['evaluation_identity'], 'reused_original': True}
                continue
            if args.reuse_only:
                waiting.append(job['id'])
                continue
            with resource_slot(ROOT / 'experiments/runs/fm_heavy_recovery_cpu.lock', config['runtime_resources'], update) as resources:
                logs = base / 'logs' / job['id']
                logs.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', '-m', 'scripts.flow_matching.evaluate_tab_ranges_sparse_job',
                           '--config', str(path), '--result', str(result)]
                env = {k.upper(): v for k, v in os.environ.items()}
                env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']))
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                with (logs / 'stdout.log').open('a', encoding='utf-8') as out, (logs / 'stderr.log').open('a', encoding='utf-8') as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
                    update({'state': 'evaluating', 'status': 'evaluating', 'active_job': job['id'], 'active_pid': child.pid,
                            'completed_jobs': len(complete), 'failed_jobs': len(failed)})
                    code, usage = guarded_wait(child, config['runtime_resources'], update)
                write_json(logs / 'resource_usage.json', dict(usage, start_resources=resources))
                final = base / 'jobs' / job['id'] / 'evaluation.json'
                if code or not final.exists():
                    failed[job['id']] = {'exit_code': code, 'resource_guard_aborted': usage['resource_guard_aborted'], 'logs_preserved': logs.relative_to(ROOT).as_posix()}
                else:
                    record = json.loads(final.read_text(encoding='utf-8'))
                    assert final.with_name('evaluation.sha256').read_text(encoding='ascii').strip() == sha(final)
                    complete[job['id']] = {'evaluation_identity': record['evaluation_identity'], 'reused_original': False}
                update({'active_pid': None, 'active_job': None})
            time.sleep(2)
        update({'status': 'waiting_training_results' if waiting else 'completed_with_unresolved_evaluations' if failed else 'completed',
                'state': 'waiting_training_results' if waiting else 'completed', 'completed': complete, 'failures': failed,
                'completed_jobs': len(complete), 'failed_jobs': len(failed), 'pending_model_results': len(waiting),
                'superseded_original_attempts': superseded})
        if args.reuse_only or not waiting:
            return
        time.sleep(config['poll_seconds'])


if __name__ == '__main__':
    main()
