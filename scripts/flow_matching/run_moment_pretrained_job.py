"""Zero-shot MOMENT on all native timestamps; weights referenced without duplication."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.model_registry import build_configured_method, load_model_config
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores, evaluate_saved, verify_job


def segment_windows(values, window):
    if len(values) < 1:
        raise ValueError('No native source points')
    if len(values) < window:
        return np.pad(values, ((0, window-len(values)), (0, 0)), mode='edge')[None], np.array([0]), window-len(values)
    windows, starts = covering_windows(values, window)
    return windows, starts, 0


def segment_scores(model, values, window):
    windows, starts, padding = segment_windows(values, window)
    raw = model.score(windows)
    score = raw[0, :len(values)] if padding else align_scores(raw, starts, len(values), window)
    if score.shape != (len(values),) or not np.isfinite(score).all():
        raise ValueError('Incomplete MOMENT native score coverage')
    return score, {'native_points': len(values), 'windows': len(windows), 'short_segment_replicated_points': padding,
                   'padding_policy': 'Right edge replication for short segments, crop scores to native points',
                   'all_context_tokens_observed': True}


def run(job, evaluation):
    import torch
    torch.set_num_threads(evaluation['cpu_threads'])
    verify_job(ROOT, job)
    output = ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('MOMENT partial artifacts preserved')
    output.mkdir(parents=True)
    config = load_model_config(ROOT / job['model_config'])
    model = build_configured_method(config, job['parameter_overrides'])
    manifest = load_model_config(ROOT / job['dataset_manifest_path'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    industrial = job['input_kind'] == 'industrial'
    if industrial:
        groups = [(role, g, g['array_prefix'] + ('_test' if role == 'test' else '_values'))
                  for role, records in [('train', dataset['training_groups']), ('validation', dataset['validation_groups']), ('test', dataset['entities'])] for g in records]
    else:
        groups = [(role, g, g['array_prefix'] + '_' + role) for g in dataset['entities'] for role in ['train', 'validation', 'test']]
    first_fit = next((g, key) for role, g, key in groups if role == 'train')
    fitting, _, _ = segment_windows(arrays[first_fit[1]], job['window'])
    start = time.monotonic()
    model.fit(fitting)
    loading_seconds = time.monotonic()-start
    checkpoint = output / 'model.pt'
    # Frozen release weights are already registered on F:; retain a verified
    # reference and strict load receipt instead of copying each model per dataset.
    torch.save({'checkpoint_reference': config['parameters']['checkpoint_directory'],
                'checkpoint_load_audit': model.checkpoint_load_audit_,
                'configuration': config, 'overrides': job['parameter_overrides']}, checkpoint)
    checkpoint_sha = sha(checkpoint)
    payload, validation, training, entities, coverage = {}, [], [], {}, []
    start = time.monotonic()
    for role, group, key in groups:
        score, audit = segment_scores(model, arrays[key], job['window'])
        prefix = group['array_prefix']
        payload[prefix+'_'+role] = score
        for suffix in ['_labels', '_timestamps', '_source_rows', '_fault_labels']:
            if prefix+suffix in arrays:
                payload[prefix+suffix] = arrays[prefix+suffix]
        coverage.append(dict(audit, role=role, group=group.get('entity_id', group.get('group_id')), source_array=key))
        if role == 'train':
            training.append(score)
        elif role == 'validation':
            validation.append(score)
        else:
            labels = arrays[prefix+'_labels']
            entities[group['entity_id']] = (labels, score)
        print(json.dumps({'id': job['id'], 'stage': 'scored_group', 'role': role, 'points': len(score)}), flush=True)
    scoring_seconds = time.monotonic()-start
    scores = output / 'scores.npz'
    np.savez_compressed(scores, **payload)
    metrics = evaluate_saved(entities, np.concatenate(validation), np.concatenate(training), dataset['input_sha256'], checkpoint_sha, evaluation, job['seed'])
    result = {'id': job['id'], 'status': 'completed', 'purpose': 'full_frozen_pretrained_zero_shot_tsad',
              'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
              'dataset': job['dataset'], 'model_config': job['model_config'], 'seed': job['seed'], 'job': job,
              'training_seconds': 0.0, 'pretrained_loading_seconds': loading_seconds, 'score_seconds': scoring_seconds,
              'parameter_count': sum(p.numel() for p in model.model.parameters()),
              'checkpoint_sha256': checkpoint_sha, 'scores_sha256': sha(scores), 'training_losses': [],
              'checkpoint_load_audit': model.checkpoint_load_audit_, 'native_coverage': coverage, 'metrics': metrics,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu'},
              'peak_cuda_allocated_bytes': None, 'paper_equivalence': False,
              'reproduction_status': config['reproduction_status'], 'boundary': config['boundary'],
              'repeat_boundary': config['repeat_boundary']}
    write_json(output / 'result.json', result)
    print(json.dumps({'id': job['id'], 'status': 'completed', 'primary_metrics': metrics['strict']['1']['micro']}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = load_model_config(args.queue)
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(job, dict(queue['evaluation'], bootstrap_block_points=max(queue['evaluation']['bootstrap_block_points'], job['window'])))


if __name__ == '__main__':
    main()
