"""Supersede unlaunched memory-heavy prototypes without changing any algorithm budget."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    original = json.loads((ROOT/'configs/experiments/fm_crossad_author_queue.v1.json').read_text())
    ablations = json.loads((ROOT/'configs/experiments/fm_crossad_ablation_queue.v1.json').read_text())
    settings = dict(original['settings'], queue_path='configs/experiments/fm_crossad_complete_queue.v3.json',
                    minimum_available_memory_bytes=24*1024**3,
                    minimum_commit_headroom_bytes=16*1024**3, emergency_commit_headroom_bytes=8*1024**3,
                    preflight_report='docs/reports/fm_crossad_complete_preflight_2026-10-09.json',
                    boundary=original['boundary']+' Reconstructed rows 1..5 explicitly distinguished. Pure encoder/decoder activation checkpointing preserves full batches and EMA single updates; low-memory starts wait.')
    common = ['src/iia_benchmark/models/fm_crossad_memory.py', 'scripts/flow_matching/run_crossad_memory_job.py',
              'scripts/flow_matching/run_crossad_complete_queue.py', 'scripts/flow_matching/register_crossad_complete.py',
              'experiments/runs/mtsad_protocol_audit/sources/crossad/snapshot.json',
              'scripts/flow_matching/heavy_resources.py',
              'docs/reports/fm_crossad_protocol_audit_2026-10-09_v2.json']
    frozen = [{'path': p, 'sha256': sha(ROOT/p)} for p in common]
    jobs = []
    for queue, runner in [(original, 'scripts/flow_matching/run_crossad_memory_job.py'), (ablations, 'scripts/flow_matching/run_crossad_ablation_job.py')]:
        for record in queue['jobs']:
            if (ROOT/record['output_directory']/'workspace').exists():
                raise ValueError('Prototype already ran; cannot silently change its runtime branch')
            job = copy.deepcopy(record)
            job.update(runner=runner, activation_checkpointing='pure_encoder_decoder', reconstructed_ablation='ablation_row' in job)
            job['frozen_files'] += frozen
            jobs.append(job)
    # Seven released checkpoints first, then original full-model baselines, then every reconstructed row.
    queue = {'schema_version': 3, 'settings': settings, 'jobs': jobs, 'boundary': settings['boundary'],
             'worker_script': 'run_crossad_complete_queue.py', 'child_script': 'run_crossad_',
             'superseded_unlaunched_queues': [original['settings']['queue_path'], ablations['settings']['queue_path'],
                                            'configs/experiments/fm_crossad_complete_queue.v2.json']}
    path = ROOT/settings['queue_path']
    if path.exists() and json.loads(path.read_text()) != queue:
        raise ValueError('CrossAD complete queue frozen')
    write_json(path, queue)
    report = {'registered_jobs': len(jobs), 'release_checkpoint_jobs': 7, 'original_budget_fresh_training_jobs': 42,
              'reconstructed_ablation_jobs': 90, 'full_row6_controls': 18, 'queue_path': settings['queue_path'],
              'queue_sha256': sha(path), 'unique_frozen_files': len({r['path'] for j in jobs for r in j['frozen_files']}),
              'superseded_unlaunched_queues': queue['superseded_unlaunched_queues'],
              'boundary': settings['boundary'], 'all_experiments_complete': False}
    write_json(ROOT/'docs/reports/fm_crossad_complete_registration_2026-10-09.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
