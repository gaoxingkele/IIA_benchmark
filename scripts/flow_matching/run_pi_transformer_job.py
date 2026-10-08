"""Execute full pinned Pi mirror budgets and save independently verifiable scores."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def binary_metrics(labels, predictions, scores=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    result = {'accuracy': float(accuracy_score(labels, predictions)), 'precision': float(p), 'recall': float(r), 'f1': float(f)}
    if scores is not None:
        result.update(auroc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                      average_precision=float(average_precision_score(labels, scores)) if np.any(labels) else None)
    return result


def author_pa(labels, predictions):
    """Keep original backward range(i,0,-1), including its index-zero boundary."""
    predicted = predictions.copy()
    state = False
    for i in range(len(labels)):
        if labels[i] == 1 and predicted[i] == 1 and not state:
            state = True
            for j in range(i, 0, -1):
                if labels[j] == 0:
                    break
                predicted[j] = 1
            for j in range(i, len(labels)):
                if labels[j] == 0:
                    break
                predicted[j] = 1
        elif labels[i] == 0:
            state = False
        if state:
            predicted[i] = 1
    return predicted


def score_controls(values, temperature, anomaly_ratio, window, multipliers, streams):
    import numpy as np
    import torch
    result = []
    labels = values['labels']
    for factor in multipliers:
        t = temperature * factor
        raw = {}
        for name in ['train', 'test']:
            mismatch = values['train_phase'] if name == 'train' else values['mismatch_test']
            mismatch = (mismatch / temperature * t).reshape(-1, window)
            rec = values['reconstruction_' + name].reshape(-1, window)
            # Match source dtype/backend for CPU reference temperature controls.
            weights = torch.softmax(-torch.from_numpy(mismatch), dim=-1).numpy()
            raw[name] = {'energy': (weights * rec).reshape(-1), 'mismatch': mismatch.reshape(-1)}
        normalized = {'train': {}, 'test': {}}
        for stream in ['energy', 'mismatch']:
            reference = raw['train'][stream]
            median = np.percentile(reference, 50)
            iqr = max(np.percentile(reference, 75) - np.percentile(reference, 25), 1e-8)
            for name in ['train', 'test']:
                normalized[name][stream] = np.maximum(0, (raw[name][stream] - median) / iqr)
        for name in ['train', 'test']:
            normalized[name]['fused'] = np.maximum(normalized[name]['energy'], normalized[name]['mismatch'])
        for stream in streams:
            train, test = normalized['train'][stream], normalized['test'][stream]
            threshold = np.percentile(np.concatenate([train, test]), 100 - anomaly_ratio)
            predicted = (test > threshold).astype(int)
            result.append({'temperature_multiplier': factor, 'temperature': t, 'stream': stream,
                           'threshold': float(threshold), 'pointwise': binary_metrics(labels, predicted, test),
                           'author_PA': binary_metrics(labels, author_pa(labels, predicted)),
                           'strict_TAB_result': False, 'test_scores_used_for_calibration': True,
                           'labels_used_for_PA': True, 'fidelity': 'Posthoc CPU score control; original author result is recorded separately'})
    return result


def run(job, settings):
    import numpy as np
    import torch
    import yaml
    for source in job['frozen_files']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen Pi source/data changed: ' + source['path'])
    output = ROOT / job['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'checkpoints').exists():
        raise ValueError('Preserve partial Pi workspace; explicit fresh retry required')
    checkpoint = output / 'checkpoints'
    checkpoint.mkdir()
    config = {'general': {'data_dir': str(ROOT / settings['raw_root']), 'device': 'cpu',
                          'model_dir': str(checkpoint),
                          'dfa_cache': str(ROOT / settings['dfa_cache_root'] / (job['id'] + '.dat'))},
              'datasets': {job['dataset']: copy.deepcopy(job['author_config'])}}
    config['datasets'][job['dataset']]['raw_directory'] = str(ROOT / job['raw_directory'])
    config_path = output / 'effective_author_config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    torch.set_num_threads(settings['cpu_threads'])
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / job['runtime']))
    from main import AnomalyDetection
    from utils.utils import set_seed
    set_seed(job['seed'])
    detector = AnomalyDetection(str(config_path), job['dataset'])
    data = job['data_audit']
    for role in ['train', 'test']:
        actual = getattr(detector.train_loader.dataset, role)
        if list(actual.shape) != data['expected_shapes'][role] or not np.isfinite(actual).all():
            raise ValueError('Incomplete/nonfinite Pi native data: ' + role)
    expected_windows = data['expected_shapes']['train'][0] - detector.win_size + 1
    if len(detector.train_loader.dataset) != expected_windows:
        raise ValueError('Author stride-one training budget changed')
    start = time.perf_counter()
    detector.train()
    training_seconds = time.perf_counter() - start
    epochs = detector.epoch_records
    expected_steps = math.ceil(expected_windows / detector.batch_size)
    if not epochs or any(e['steps'] != expected_steps for e in epochs):
        raise ValueError('Incomplete full author training steps')
    if len(epochs) != detector.n_epochs and not epochs[-1]['early_stop']:
        raise ValueError('Short Pi epoch budget without author early stopping')
    model_path = checkpoint / (job['dataset'] + '_checkpoint.pth')
    weights = torch.load(model_path, map_location='cpu', weights_only=True)
    if not weights or any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Invalid/nonfinite Pi checkpoint')
    start = time.perf_counter()
    with torch.no_grad():
        returned = detector.test()
    evaluation_seconds = time.perf_counter() - start
    score_path = checkpoint / 'author_scores.npz'
    with np.load(score_path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    native = detector.thre_loader.dataset.test_labels.reshape(-1).astype(int)
    expected_points = len(detector.thre_loader.dataset) * detector.win_size
    if len(values['labels']) != expected_points or not np.array_equal(values['labels'], native[:expected_points]):
        raise ValueError('Author scored labels/coverage differs')
    for key, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite Pi saved stream: ' + key)
    if len(values['train_energy']) != expected_windows * detector.win_size:
        raise ValueError('Incomplete full overlapping train scores')
    threshold = np.percentile(np.concatenate([values['fused_train'], values['test_energy']]), 100-detector.anomaly_ratio)
    if threshold != values['threshold'] or not np.array_equal(values['raw_predictions'], values['test_energy'] > threshold):
        raise ValueError('Author threshold cannot be reproduced')
    if not np.array_equal(author_pa(values['labels'], values['raw_predictions']), values['pa_predictions']):
        raise ValueError('Original author PA differs')
    metrics = binary_metrics(values['labels'], values['pa_predictions'])
    if any(not math.isclose(metrics[k], float(v), rel_tol=1e-12, abs_tol=1e-12) for k, v in zip(['accuracy','precision','recall','f1'], returned)):
        raise ValueError('Reported Pi metrics differ from saved scores')
    metrics['pointwise_no_PA_same_author_threshold'] = binary_metrics(values['labels'], values['raw_predictions'], values['test_energy'])
    controls = score_controls(values, detector.temperature, detector.anomaly_ratio, detector.win_size,
                              settings['score_temperature_multipliers'], settings['score_streams'])
    control_path = output / 'posthoc_score_controls.json'
    write_json(control_path, {'records': controls, 'boundary': settings['boundary']})
    paths = [config_path, model_path, score_path, checkpoint / 'epochs.json', checkpoint / (job['dataset'] + '_global_hurst.pt'), control_path,
             (ROOT / settings['dfa_cache_root'] / (job['id'] + '.json'))]
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    receipt = {'status': 'completed', 'experiment_sha256': fingerprint, 'metrics': metrics,
               'native_train_points': data['expected_shapes']['train'][0], 'training_windows': expected_windows,
               'steps_per_epoch': expected_steps, 'epochs': epochs, 'native_test_points': len(native),
               'scored_test_points': expected_points, 'dropped_test_tail_points': len(native)-expected_points,
               'active_attention_layers': len(detector.model.encoder.attn_layers),
               'parameter_count': sum(p.numel() for p in detector.model.parameters()),
               'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
               'torch_version': torch.__version__, 'strict_TAB_result': False,
               'test_used_for_early_stopping': True, 'test_scores_used_for_calibration': True,
               'artifacts': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths],
               'boundary': settings['boundary']}
    write_json(output / 'result.json', receipt)
    print(json.dumps({'status': 'completed', 'id': job['id'], 'metrics': metrics}))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(job, queue['settings'])


if __name__ == '__main__':
    main()
