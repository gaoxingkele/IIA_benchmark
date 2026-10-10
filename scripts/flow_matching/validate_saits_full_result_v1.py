"""Validate complete SAITS/BRITS checkpoints and every held-out unit's statistics."""
import argparse
from configparser import ConfigParser
import json
import math
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest


def validate(config_path):
    import h5py
    import numpy as np
    import torch
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('diagnostic') or config.get('limit_test_cases'):
        raise ValueError('A shortened/diagnostic job cannot satisfy original completion')
    output = ROOT / config['output_root']
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed' or result['config_sha256'] != digest(config_path) or result['config'] != config:
        raise ValueError('Full original job/result identity differs')
    author_config = ROOT / config['author_config']
    if digest(author_config) != config['author_config_sha256']:
        raise ValueError('Frozen full author hyperparameters changed')
    data = ROOT / config['dataset_root']
    audit = json.loads((data / 'data_audit.json').read_text(encoding='utf-8'))
    dataset_sha = digest(data / 'datasets.h5')
    if dataset_sha != config['dataset_sha256'] or dataset_sha != result['dataset_sha256'] or dataset_sha != audit['dataset_sha256']:
        raise ValueError('Full original processed dataset binding differs')
    if digest(data / 'audit_arrays.npz') != audit['audit_arrays_sha256']:
        raise ValueError('Frozen test identity arrays differ')
    checkpoint = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
    if (checkpoint['config_sha256'] != digest(config_path) or checkpoint['dataset_sha256'] != dataset_sha
            or not checkpoint['finished']):
        raise ValueError('Full original resumed training did not finish')
    parameters = ConfigParser(); parameters.read(author_config, encoding='utf-8')
    maximum = parameters.getint('training', 'epochs')
    if not 0 < checkpoint['next_epoch'] <= maximum:
        raise ValueError('Original epoch budget is invalid')
    if checkpoint['next_epoch'] < maximum and checkpoint['patience'] > 0:
        raise ValueError('Early stopping must exhaust original patience')
    required_rng = {'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'}
    if not required_rng.issubset(checkpoint) or not checkpoint['optimizer']['state']:
        raise ValueError('Complete optimizer/RNG state missing')
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    expected = {k: v for k, v in best.items() if k != 'model_state_dict'}
    if expected != result['validation_best'] or not best['model_state_dict']:
        raise ValueError('Validation-selected checkpoint binding differs')
    if not all(torch.isfinite(v).all() for v in best['model_state_dict'].values()):
        raise ValueError('Selected checkpoint contains nonfinite weights')
    statistic_name = 'group_statistics.npz' if config.get('bootstrap_group') == 'month' else 'patient_statistics.npz'
    with np.load(data / 'audit_arrays.npz', allow_pickle=False) as identity, np.load(output / statistic_name, allow_pickle=False) as measured:
        wanted = identity['test_ids']; actual = measured['unit_ids']; statistics = measured['statistics']
        if sorted(wanted.tolist()) != sorted(actual.tolist()) or len(np.unique(actual)) != len(actual):
            raise ValueError('Full held-out unit coverage differs')
        groups = np.unique(identity['test_groups']) if config.get('bootstrap_group') == 'month' else np.unique(wanted)
        if not np.array_equal(np.sort(groups), np.sort(measured['group_ids'])):
            raise ValueError('Grouped uncertainty identities differ')
        if statistics.shape != (len(groups), 4) or not np.isfinite(statistics).all():
            raise ValueError('Complete grouped error statistics missing')
        total = statistics.sum(axis=0)
    with h5py.File(data / 'datasets.h5', 'r') as stream:
        mask = stream['test']['indicating_mask']
        points = sum(float(mask[start:start+128].sum()) for start in range(0, len(mask), 128))
        if len(mask) != len(wanted):
            raise ValueError('Full native test data row count differs')
    if points != result['target_count'] or total[2] != points or points <= 0:
        raise ValueError('Incomplete held-out coordinate coverage')
    metrics = dict(mae=float(total[0]/total[2]), rmse=math.sqrt(float(total[1]/total[2])),
                   mre=float(total[0]/total[3]))
    for key, value in metrics.items():
        if not math.isclose(value, result['metrics'][key], rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError('Full grouped independent metric differs: ' + key)
    return dict(id=config['id'], full_training_finished=True, full_test_units=len(wanted),
        held_out_coordinates=points, metrics_independently_recomputed=True,
        original_maximum_epochs=maximum, next_epoch=checkpoint['next_epoch'],
        author_patience_remaining=checkpoint['patience'], benchmark_smoke=False,
        result_sha256=digest(output/'result.json'), resume_sha256=digest(output/'resume.pt'),
        best_sha256=digest(output/'best.pt'), statistics_sha256=digest(output/statistic_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(ROOT / args.config)))
