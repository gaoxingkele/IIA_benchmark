"""Train once on unique industrial fit groups and score every held-out point."""
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
from scripts.flow_matching.model_registry import build_configured_method, load_model_config
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores, evaluate_saved, verify_job


def pooled_training_windows(data, dataset, window):
    groups = dataset['training_groups']
    prefixes = [g['array_prefix'] for g in groups]
    if not groups or len(set(prefixes)) != len(prefixes):
        raise ValueError('Repeated or missing training group prefixes')
    windows, budget = [], []
    for g in groups:
        values = data[g['array_prefix'] + '_values']
        if len(values) != g['points']:
            raise ValueError('Incomplete industrial fit group')
        batch, _ = covering_windows(values, window)
        windows.append(batch)
        budget.append({'group_id': g['group_id'], 'source_interval': g['source_interval'], 'points': len(values), 'windows': len(batch)})
    pooled = np.concatenate(windows)
    if sum(b['points'] for b in budget) != dataset['scaler']['fit_points']:
        raise ValueError('Industrial training was duplicated or truncated')
    return pooled, budget


def verify_training_and_checkpoint(model, configuration, modules):
    import torch
    if not modules or any(not torch.isfinite(t).all() for state in modules.values() for t in state.values() if t.is_floating_point()):
        raise ValueError('Missing or nonfinite industrial checkpoint')
    losses = getattr(model, 'training_losses_', getattr(model, 'loss_history_', None))
    parameters = configuration['parameters']
    expected = parameters['epochs'] + (parameters.get('reflow_stages', 1)-1)*parameters.get('reflow_epochs', 0)
    if losses is not None and (len(losses) != expected or not np.isfinite(losses).all()):
        raise ValueError('Incomplete or nonfinite full training epoch history')
    return losses, {'declared_epochs': expected, 'recorded_epochs': len(losses) if losses is not None else None,
                    'history_boundary': 'Methods without epoch histories retain configured full loops in frozen implementation; absence is explicit, not evidence of measured epoch coverage'}


def run(root, job, settings):
    import torch
    torch.set_num_threads(settings['cpu_threads'])
    verify_job(root, job)
    output = root/job['output_directory']
    if output.exists():
        raise FileExistsError('Existing industrial artifacts preserved')
    output.mkdir(parents=True)
    configuration = load_model_config(root/job['model_config'])
    model = build_configured_method(configuration, job['parameter_overrides'])
    manifest = load_model_config(root/job['dataset_manifest_path'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    with np.load(root/dataset['input_npz'], allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    window = job['window']
    training, budget = pooled_training_windows(data, dataset, window)
    print(json.dumps({'id': job['id'], 'stage': 'training', 'windows': len(training), 'unique_training_groups': len(budget),
                      'epochs': configuration['parameters']['epochs'], 'device': job['parameter_overrides']['device']}), flush=True)
    uses_cuda = str(job['parameter_overrides']['device']).startswith('cuda')
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.fit(training)
    training_seconds = time.monotonic()-started
    modules = {name: module.state_dict() for name, module in vars(model).items() if isinstance(module, torch.nn.Module)}
    losses, epoch_audit = verify_training_and_checkpoint(model, configuration, modules)
    checkpoint = output/'model.pt'
    torch.save({'modules': modules, 'configuration': configuration, 'overrides': job['parameter_overrides'],
                'adjacency': getattr(model, 'adjacency_', None), 'window': window, 'unique_training_groups': budget}, checkpoint)
    checkpoint_sha = sha(checkpoint)
    payload, validation, train_scores, entities = {}, [], [], {}
    score_started = time.monotonic()
    for role, groups in [('train', dataset['training_groups']), ('validation', dataset['validation_groups']), ('test', dataset['entities'])]:
        for g in groups:
            prefix = g['array_prefix']
            values = data[prefix + ('_test' if role == 'test' else '_values')]
            if len(values) != g['points']:
                raise ValueError('Industrial score input was truncated')
            windows, starts = covering_windows(values, window)
            scores = align_scores(model.score(windows), starts, len(values), window)
            payload[prefix+'_'+role] = scores
            for suffix in ('_labels', '_timestamps', '_source_rows', '_fault_labels'):
                if prefix+suffix in data:
                    payload[prefix+suffix] = data[prefix+suffix]
            if role == 'train':
                train_scores.append(scores)
            elif role == 'validation':
                validation.append(scores)
            else:
                labels = data[prefix+'_labels']
                if labels.shape != scores.shape or len(labels) != g['test_points']:
                    raise ValueError('Industrial full native test coverage failed')
                entities[g['entity_id']] = (labels, scores)
            print(json.dumps({'id': job['id'], 'stage': 'scored_group', 'role': role, 'entity': g['entity_id'], 'points': len(scores)}), flush=True)
    score_seconds = time.monotonic()-score_started
    score_file = output/'scores.npz'
    np.savez_compressed(score_file, **payload)
    metrics = evaluate_saved(entities, np.concatenate(validation), np.concatenate(train_scores), dataset['input_sha256'], checkpoint_sha, settings, job['seed'])
    result = {'id': job['id'], 'status': 'completed', 'purpose': 'full_grouped_industrial_tsad_transfer',
              'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
              'dataset': job['dataset'], 'model_config': job['model_config'], 'seed': job['seed'], 'job': job,
              'training_seconds': training_seconds, 'score_seconds': score_seconds,
              'parameter_count': sum(p.numel() for m in vars(model).values() if isinstance(m, torch.nn.Module) for p in m.parameters()),
              'checkpoint_sha256': checkpoint_sha, 'scores_sha256': sha(score_file), 'training_losses': losses, 'metrics': metrics,
              'fit_group_budget': budget, 'epoch_audit': epoch_audit, 'unique_fit_windows': len(training),
              'test_points': sum(len(pair[0]) for pair in entities.values()), 'test_entities': len(entities),
              'validation_points': sum(len(s) for s in validation), 'selection_audit': dataset['selection_audit'],
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                              'cuda': torch.version.cuda, 'device': job['parameter_overrides']['device']},
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if uses_cuda else None,
              'paper_equivalence': False, 'reproduction_status': configuration['reproduction_status'],
              'boundary': 'Frozen whole-run/day industrial transfer. Shared fit groups trained once, every test timestamp scored. Validation-only nominal tail quantiles, no PA. SKAB calibration includes native anomalies; PRONTO normal-only fit/validation selected with native labels; days/rotations correlated. PSM baseline presets transferred without target tuning. Original-paper and full TAB harness equivalence remain separate.'}
    write_json(output/'result.json', result)
    print(json.dumps({'id': job['id'], 'stage': 'completed', 'primary_metrics': metrics['strict'][str(settings['primary_false_alarm_percent'])]['micro']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = load_model_config(args.queue)
    run(ROOT, next(j for j in queue['jobs'] if j['id'] == args.job_id), queue['evaluation'])


if __name__ == '__main__':
    main()
