"""Follow an additional frozen range-metric queue without replacing live v1."""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for record in config['source_receipts']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Pinned metric source changed: ' + record['path'])
    lock = exclusive_lock(ROOT / config['output_root'] / 'evaluation.lock')
    if lock is None:
        raise RuntimeError('Additional TAB evaluator already has a live owner')
    reference = load_reference(ROOT / config['reference_metric_root'])
    jobs = [job for path in config['queue_paths'] for job in json.loads((ROOT / path).read_text(encoding='utf-8'))['jobs']]
    state_path = ROOT / config['output_root'] / 'status.json'
    completed, failures = {}, {}
    while True:
        pending = []
        for job in jobs:
            if job['id'] in completed or job['id'] in failures:
                continue
            result = ROOT / job['output_directory'] / 'result.json'
            if not result.exists():
                pending.append(job['id'])
                continue
            write_json(state_path, {'status': 'evaluating', 'pid': os.getpid(), 'active_job': job['id'],
                                    'completed_jobs': len(completed), 'failed_jobs': len(failures)})
            try:
                record = evaluate(ROOT, result, config, reference)
                completed[job['id']] = {'evaluation_identity': record['evaluation_identity']}
            except Exception as error:
                failures[job['id']] = {'error_type': type(error).__name__, 'error': str(error)}
        write_json(state_path, {'status': 'waiting_training_results' if pending else 'completed_with_unresolved_evaluations' if failures else 'completed',
                                'pid': os.getpid(), 'completed_jobs': len(completed), 'failed_jobs': len(failures),
                                'pending_model_results': len(pending), 'completed': completed, 'failures': failures})
        if not pending:
            return 1 if failures else 0
        time.sleep(config['poll_seconds'])


if __name__ == '__main__':
    raise SystemExit(main())
