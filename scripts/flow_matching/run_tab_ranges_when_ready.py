"""Evaluate every completed registered TSAD job once and follow live queues."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock
from scripts.flow_matching.tab_range_metrics import load_reference
from scripts.flow_matching.evaluate_tab_ranges import evaluate


def main():
    path = ROOT / 'configs/experiments/fm_tab_range_execution.v1.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    for record in config['source_receipts']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Pinned TAB evaluator source changed')
    lock = exclusive_lock(ROOT / config['output_root'] / 'evaluation.lock')
    if lock is None:
        raise RuntimeError('A TAB range evaluation continuation is already live')
    reference = load_reference(ROOT / config['reference_metric_root'])
    jobs = []
    for queue_path in config['queue_paths']:
        queue = json.loads((ROOT / queue_path).read_text(encoding='utf-8'))
        jobs.extend(queue['jobs'])
    state_path = ROOT / config['output_root'] / 'status.json'
    completed, failures = {}, {}
    while True:
        pending = []
        superseded = {job['supersedes_failed_numerics_job']: job['id'] for job in jobs
                      if job.get('supersedes_failed_numerics_job') and job['id'] in completed}
        for job in jobs:
            if job['id'] in completed or job['id'] in failures or job['id'] in superseded:
                continue
            result = ROOT / job['output_directory'] / 'result.json'
            if not result.exists():
                pending.append(job['id'])
                continue
            write_json(state_path, {'status': 'evaluating', 'pid': os.getpid(), 'active_job': job['id'],
                                    'completed_jobs': len(completed), 'failed_jobs': len(failures), 'registered_attempts': len(jobs)})
            try:
                record = evaluate(ROOT, result, config, reference)
                completed[job['id']] = {'evaluation_path': (ROOT / config['output_root'] / 'jobs' / job['id'] / 'evaluation.json').relative_to(ROOT).as_posix(),
                                         'evaluation_identity': record['evaluation_identity']}
            except Exception as error:
                failures[job['id']] = {'error_type': type(error).__name__, 'error': str(error)}
        status = 'waiting_training_results' if pending else 'completed' if not failures else 'completed_with_unresolved_evaluations'
        write_json(state_path, {'status': status, 'pid': os.getpid(), 'completed_jobs': len(completed), 'failed_jobs': len(failures),
                                'registered_attempts': len(jobs), 'pending_model_results': len(pending),
                                'completed': completed, 'failures': failures, 'superseded_original_attempts': superseded})
        if not pending:
            return 0 if not failures else 1
        time.sleep(config['poll_seconds'])


if __name__ == '__main__':
    raise SystemExit(main())
