"""Train a frozen local method on full TSAD data; save scores and strict metrics."""
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
from iia_benchmark.evaluation.mtsad_strict import calibrate_validation, evaluate_entities
from iia_benchmark.evaluation.mtsad_metrics import anomaly_ranges, pointwise_metrics, point_adjust


def covering_windows(values, window):
    """Nonoverlapping windows plus one end-aligned tail; no padding or omitted points."""
    if values.ndim != 2 or len(values) < window or window < 1:
        raise ValueError('Series shorter than a window')
    starts = np.arange(0, len(values) - window + 1, window, dtype=np.int64)
    if starts[-1] != len(values) - window:
        starts = np.append(starts, len(values) - window)
    return values[starts[:, None] + np.arange(window)], starts


def align_scores(scores, starts, length, window):
    scores = np.asarray(scores, dtype=float)
    if scores.shape == (len(starts),):
        scores = np.repeat(scores[:, None], window, axis=1)
    if scores.shape != (len(starts), window) or not np.isfinite(scores).all():
        raise ValueError('Incorrect/nonfinite window score shape')
    indexes = starts[:, None] + np.arange(window)
    totals, counts = np.zeros(length), np.zeros(length, dtype=np.int64)
    np.add.at(totals, indexes.ravel(), scores.ravel())
    np.add.at(counts, indexes.ravel(), 1)
    if not counts.all():
        raise ValueError('Incomplete test timestamp coverage')
    return totals / counts


def block_f1_interval(entities, threshold, seed, repeats, block_length):
    """Cluster bootstrap for multiple entities; temporal blocks for a single entity."""
    if repeats < 2 or block_length < 1:
        raise ValueError('Invalid resampling settings')
    rng = np.random.default_rng(seed)
    series = list(entities.values())
    values = []
    for _ in range(repeats):
        if len(series) > 1:
            sampled = [series[i] for i in rng.integers(0, len(series), len(series))]
            truth, score = [np.concatenate([pair[i] for pair in sampled]) for i in (0, 1)]
        else:
            original_truth, original_score = series[0]
            starts = rng.integers(0, len(original_truth), int(np.ceil(len(original_truth) / block_length)))
            indexes = ((starts[:, None] + np.arange(block_length)) % len(original_truth)).ravel()[:len(original_truth)]
            truth, score = original_truth[indexes], original_score[indexes]
        values.append(pointwise_metrics(truth, score > threshold)['f1'])
    low, high = np.percentile(values, [2.5, 97.5])
    return {'low': float(low), 'high': float(high), 'resamples': repeats, 'seed': seed,
            'unit': 'entity_cluster' if len(series) > 1 else 'circular_temporal_block',
            'block_points': block_length if len(series) == 1 else None,
            'boundary': 'Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty.'}


def event_metrics(entities, threshold):
    events, detected, delays, normal, false = 0, 0, [], 0, 0
    for truth, scores in entities.values():
        prediction = scores > threshold
        normal += int((~truth.astype(bool)).sum())
        false += int((prediction & ~truth.astype(bool)).sum())
        for start, end in anomaly_ranges(truth):
            events += 1
            hits = np.flatnonzero(prediction[start:end + 1])
            if len(hits):
                detected += 1
                delays.append(int(hits[0]))
    return {'event_recall': detected / events if events else None, 'events': events, 'detected_events': detected,
            'mean_detection_delay_points': float(np.mean(delays)) if delays else None,
            'false_alarms_per_1000_normal_points': 1000 * false / normal if normal else None,
            'delay_boundary': 'Offline window scores use future context; delay in source points, not causal latency. Missed events reported separately.'}


def evaluate_saved(entities, validation, training, input_sha, checkpoint_sha, settings, seed):
    expected = {key: len(pair[0]) for key, pair in entities.items()}
    strict = {}
    for ratio in settings['strict_false_alarm_percent']:
        calibration = calibrate_validation(validation, false_alarm_percent=ratio, calibration_split='validation',
                                            calibration_dataset_sha256=input_sha, model_artifact_sha256=checkpoint_sha)
        record = evaluate_entities(entities, expected_lengths=expected, calibration=calibration)
        record['event_metrics'] = event_metrics(entities, calibration.threshold)
        record['f1_ci95'] = block_f1_interval(entities, calibration.threshold, seed + 10000,
                                             settings['bootstrap_resamples'], settings['bootstrap_block_points'])
        strict[str(ratio)] = record
    # Same aligned scores; disclose pooled test-score calibration and test-label selection.
    # This is TAB's core metric/ratio diagnostic, not its full training/VUS/Affiliation harness.
    grid = []
    all_scores = np.concatenate([pair[1] for pair in entities.values()])
    labels = np.concatenate([pair[0] for pair in entities.values()])
    for ratio in settings['tab_threshold_ratios_percent']:
        threshold = float(np.percentile(np.concatenate([training, all_scores]), 100 - ratio))
        pred = all_scores > threshold
        adjusted = np.concatenate([point_adjust(truth, scores > threshold) for truth, scores in entities.values()])
        grid.append({'ratio_percent': ratio, 'threshold': threshold,
                     'f_score': pointwise_metrics(labels, pred)['f1'],
                     'adjust_f_score': pointwise_metrics(labels, adjusted)['f1']})
    primary = strict[str(settings['primary_false_alarm_percent'])]['micro']
    return {'strict': strict, 'tab_core_diagnostic': {
        'tab_commit': settings['tab_commit'], 'ratio_grid': grid,
        'best_grid_f_score': max(row['f_score'] for row in grid),
        'best_grid_adjust_f_score': max(row['adjust_f_score'] for row in grid),
        'auc_roc': primary['auroc'], 'auc_pr': primary['average_precision'],
        'test_scores_used_in_calibration': True, 'test_labels_used_to_select_best_grid': True,
        'pending_metrics': ['VUS_ROC', 'VUS_PR', 'affiliation_f'],
        'boundary': 'Core TAB-equivalent metric formulas on local saved scores; no full TAB or author reproduction claim.'}}


def verify_job(root, job):
    for record in job['frozen_files']:
        if sha(root / record['path']) != record['sha256']:
            raise ValueError('Frozen experiment source changed: ' + record['path'])


def run(root, job, settings):
    import torch
    torch.set_num_threads(settings['cpu_threads'])
    verify_job(root, job)
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    output = root / job['output_directory']
    if output.exists():
        raise FileExistsError('Existing experiment artifacts preserved: ' + str(output))
    output.mkdir(parents=True)
    model_config = load_model_config(root / job['model_config'])
    model = build_configured_method(model_config, job['parameter_overrides'])
    window = job['window']
    manifest = load_model_config(root / job['dataset_manifest_path'])
    dataset = next(item for item in manifest['datasets'] if item['id'] == job['dataset'])
    with np.load(root / dataset['input_npz'], allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    training = np.concatenate([covering_windows(data[e['array_prefix'] + '_train'], window)[0] for e in dataset['entities']])
    print(json.dumps({'id': job['id'], 'stage': 'training', 'windows': len(training), 'epochs': model_config['parameters']['epochs'],
                      'device': job['parameter_overrides']['device']}), flush=True)
    if str(job['parameter_overrides']['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model.fit(training)
    training_seconds = time.monotonic() - start
    modules = {name: module.state_dict() for name, module in vars(model).items() if isinstance(module, torch.nn.Module)}
    if not modules:
        raise ValueError('Fitted method exposes no checkpoint modules')
    checkpoint = output / 'model.pt'
    torch.save({'modules': modules, 'configuration': model_config, 'overrides': job['parameter_overrides'],
                'adjacency': getattr(model, 'adjacency_', None), 'window': window}, checkpoint)
    checkpoint_sha = sha(checkpoint)
    losses = getattr(model, 'training_losses_', getattr(model, 'loss_history_', None))
    payload, validation, train_scores, entities = {}, [], [], {}
    score_start = time.monotonic()
    for entity in dataset['entities']:
        prefix = entity['array_prefix']
        for name in ('train', 'validation', 'test'):
            values = data[prefix + '_' + name]
            windows, starts = covering_windows(values, window)
            raw = model.score(windows)
            score = align_scores(raw, starts, len(values), window)
            payload[prefix + '_' + name] = score
            if name == 'train':
                train_scores.append(score)
            elif name == 'validation':
                validation.append(score)
            else:
                labels = data[prefix + '_labels']
                payload[prefix + '_labels'] = labels
                entities[entity['entity_id']] = (labels, score)
        print(json.dumps({'id': job['id'], 'stage': 'scored_entity', 'entity': entity['entity_id']}), flush=True)
    score_seconds = time.monotonic() - score_start
    score_file = output / 'scores.npz'
    np.savez_compressed(score_file, **payload)
    metrics = evaluate_saved(entities, np.concatenate(validation), np.concatenate(train_scores), dataset['input_sha256'],
                              checkpoint_sha, settings, job['seed'])
    parameter_count = sum(p.numel() for module in vars(model).values() if isinstance(module, torch.nn.Module) for p in module.parameters())
    result = {'id': job['id'], 'status': 'completed', 'purpose': 'full_local_tsad_benchmark', 'experiment_sha256': fingerprint,
              'dataset': job['dataset'], 'model_config': job['model_config'], 'seed': job['seed'], 'job': job,
              'training_seconds': training_seconds, 'score_seconds': score_seconds,
              'parameter_count': parameter_count, 'checkpoint_sha256': checkpoint_sha, 'scores_sha256': sha(score_file),
              'training_losses': losses, 'metrics': metrics,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                              'cuda': torch.version.cuda, 'device': job['parameter_overrides']['device']},
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if str(job['parameter_overrides']['device']).startswith('cuda') else None,
              'paper_equivalence': False, 'reproduction_status': model_config['reproduction_status'],
              'boundary': 'Full saved-score local adaptation/reconstruction benchmark. Registered training defaults retained; independently purged validation calibration; offline full timestamp coverage. Original-paper fidelity and full TAB range metrics remain separate requirements.'}
    write_json(output / 'result.json', result)
    print(json.dumps({'id': job['id'], 'stage': 'completed', 'primary_metrics': metrics['strict'][str(settings['primary_false_alarm_percent'])]['micro']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = load_model_config(args.queue)
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(ROOT, job, queue['evaluation'])


if __name__ == '__main__':
    main()
