"""Inspect an immutable full native checkpoint after terminal sampling failure.

No model forward, altered sampler, evaluator, or performance claim is produced.
The complete canonical job is frozen rather than replacing its failed attempt.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha, write
from scripts.flow_matching.audit_spectral_preserved_sampling_failures_v1 import checkpoint_budget
from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity


def selected_job(queue, config):
    jobs = queue['jobs']
    if len(jobs) != 70 or len({job['id'] for job in jobs}) != 70:
        raise ValueError('All fourteen original five-seed cohorts must remain registered')
    matches = [job for job in jobs if job['id'] == config['original_job']['id']]
    if len(matches) != 1 or matches[0] != config['original_job']:
        raise ValueError('Every field of the frozen full canonical job must match')
    job = matches[0]
    if job['model'] != 'spectral_flow' or job['generation_field_chunk_size'] is not None:
        raise ValueError('Unmodified full original Spectral sampling required')
    return job


def terminal_failure(job, progress, receipt, result_exists, generated_exists):
    if (receipt['id'] != job['id'] or receipt['exitcode'] == 0
            or receipt['full_result_validated'] or receipt['resource_guard_aborted'] is not True
            or progress != {'phase': 'sampling', 'optimizer_updates': job['training_updates']}
            or result_exists or generated_exists):
        raise ValueError('Terminal resource failure after full training, before full samples required')


def audit(config, config_path):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen original evidence changed: ' + item['path'])
    queue = read(ROOT / config['queue'])
    job = selected_job(queue, config)
    native = config['terminal_native_handle']
    handle = process_identity(native['pid'], native['create_time'], native['command_token'])
    if handle['live']:
        raise ValueError('Original native worker is still live; preserve ongoing work')
    output = ROOT / job['output_directory']
    artifact = Path(job['artifact_directory'])
    receipt_path = ROOT / queue['state_root'] / 'receipts' / (job['id'] + '.json')
    receipt = read(receipt_path)
    progress = read(output / 'progress.json')
    terminal_failure(job, progress, receipt, (output / 'result.json').exists(),
                     (artifact / 'fake_data.npy').exists())
    import numpy as np
    import torch
    torch.set_num_threads(config['cpu_threads'])
    checkpoint = artifact / 'author_output/checkpoints/checkpoint-10.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    budget = checkpoint_budget(saved, job['training_updates'], torch)
    data = []
    for role, path, expected in [('training', job['frozen_samples'], job['data_sha256']),
                                 ('reference', job['reference_path'], job['reference_sha256'])]:
        if sha(ROOT / path) != expected:
            raise ValueError('Full original data changed: ' + path)
        values = np.load(ROOT / path, mmap_mode='r', allow_pickle=False)
        if list(values.shape) != job['data_shape'] or not np.isfinite(values).all():
            raise ValueError('Full original finite data shape required')
        data.append(dict(role=role, path=path, sha256=expected, shape=list(values.shape)))
    if torch.cuda.is_initialized():
        raise ValueError('Read-only checkpoint inspection must not initialize CUDA')
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=sha(ROOT / config_path), full_original_registered_jobs=70,
        full_original_registered_cohorts=14, original_job=job,
        actual_terminal_native_handle=handle, actual_original_progress=progress,
        actual_terminal_resource_receipt=receipt, receipt_sha256=sha(receipt_path),
        full_training_checkpoint_path=str(checkpoint), full_training_checkpoint_sha256=sha(checkpoint),
        original_complete_data=data, actual_failed_full_training_checkpoints_verified=1,
        actual_finished_generation_experiments=0, additional_training_seed_slots=0,
        cuda_context_initialized=False, model_reinference_executed=False,
        complete_generated_array_available=False, complete_four_evaluator_metrics_available=False,
        author_source_or_scientific_sampling_budget_changed=False,
        author_equivalence_certified=False, all_original_experiments_complete=False,
        **budget, boundary=config['boundary'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable checkpoint inspection')
    proof = audit(config, args.config)
    target.mkdir(parents=True)
    write(target / 'full_native_checkpoint_inspection.json', proof)
    write(target / 'validation.json', dict(passed=True, additional_training_seed_slots=0,
        source_receipts=config['source_receipts'] + [dict(path=args.config, sha256=sha(ROOT / args.config))],
        outputs={'full_native_checkpoint_inspection.json': sha(target / 'full_native_checkpoint_inspection.json')}))
    print(dict(full_training_checkpoint_verified=True, full_sampling_complete=False,
               exact_sampling_RNG_resume_certified=False, performance_result_available=False))


if __name__ == '__main__':
    main()
