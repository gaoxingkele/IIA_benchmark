"""Run patched author CFMI CLI, retaining author training/evaluation configs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summarize_author_metric(values, key):
    """Match author's nanmean for per-coordinate CRPS, never hide infinities."""
    values = np.asarray(values)
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError(f'Invalid author metric: {key}')
    undefined = int(np.isnan(values).sum())
    if undefined and key != 'crps':
        raise ValueError(f'Non-finite author metric: {key}')
    return float(np.nanmean(values)), {'components': int(values.size), 'undefined_components': undefined}


def latest_checkpoint(training_root):
    paths = list(training_root.glob('train/seed_m*_d*/lightning_logs/version_*/checkpoints/last.ckpt'))
    return max(paths, key=lambda p: int(p.parents[1].name.removeprefix('version_'))) if paths else None


def checkpoint_complete(checkpoint, source, epochs):
    if checkpoint is None:
        return False
    # Author Lightning checkpoints pickle the vector-field class, not just tensors.
    # This trusted local snapshot must be importable before deserialization.
    sys.path.insert(0, str(source.resolve()))
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return int(state['epoch']) + 1 >= epochs


def execute(command, output, label):
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    with (output / f'{label}.stdout.log').open('a', encoding='utf-8') as out, (output / f'{label}.stderr.log').open('a', encoding='utf-8') as err:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err, check=True)


def run(config_path):
    settings = json.loads(Path(config_path).read_text(encoding='utf-8'))
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    for key in ['train_config', 'eval_config', 'data_root']:
        expected = settings.get('processed_sha256' if key == 'data_root' else key + '_sha256')
        if expected and hashlib.sha256((ROOT / settings[key]).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {key}')
    output = ROOT / settings['output_root']
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['config_sha256'] != config_hash:
            raise ValueError('Completed result uses a different configuration')
        return result
    source = ROOT / settings['author_source']
    training_root = ROOT / settings['training_root']
    training_root.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(training_root)
    import torch
    complete = checkpoint_complete(checkpoint, source, settings['epochs'])
    status = {'id': settings['id'], 'status': 'running', 'pid': os.getpid(), 'config': settings,
              'config_sha256': config_hash, 'author_snapshot': json.loads((source.parent / 'snapshot.json').read_text()),
              'patch_manifest': json.loads((source.parent / 'patch_manifest.json').read_text()),
              'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}}
    (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    started = time.monotonic()
    if not complete:
        train_config = ROOT / settings['train_config'] if settings.get('train_config') else source / settings['author_train_config']
        command = [sys.executable, str(source / 'train.py'), '--config', str(train_config),
                   '--data.dataset.init_args.root', str(ROOT / settings['data_root']),
                   '--trainer.default_root_dir', str(training_root), '--experiment_subdir_base', 'train',
                   '--trainer.accelerator', 'gpu', '--trainer.devices', '1', '--trainer.max_epochs', str(settings['epochs']),
                   '--trainer.enable_progress_bar', 'false']
        if checkpoint:
            command += ['--resume_checkpoint', str(checkpoint)]
        execute(command, output, 'train')
        checkpoint = latest_checkpoint(training_root)
        if checkpoint is None:
            raise FileNotFoundError('Author trainer did not create last.ckpt')
    status['training_seconds'] = time.monotonic() - started
    status['checkpoint'] = str(checkpoint.relative_to(ROOT))
    started = time.monotonic()
    command = [sys.executable, str(source / 'eval_imputation_timeseries.py'), '--config', str(source / settings.get('author_eval_config', '')),
               '--data.dataset.init_args.root', str(ROOT / settings['data_root']), '--cfm_model_path', str(checkpoint),
               '--default_root_dir', str(output), '--experiment_subdir_base', 'evaluation',
               '--num_imputations', str(settings['num_samples'])]
    if settings.get('metric_protocol'):
        command = [sys.executable, str(ROOT / 'scripts/flow_matching/evaluate_cfmi_transfer.py'),
                   '--campaign-config', str(Path(config_path).resolve()), '--checkpoint', str(checkpoint)]
    elif settings.get('dual_air36_cutoff'):
        command = [sys.executable, str(ROOT / 'scripts/flow_matching/evaluate_cfmi_air36.py'),
                   '--campaign-config', str(Path(config_path).resolve()), '--checkpoint', str(checkpoint)]
    execute(command, output, 'evaluate')
    metrics, metric_audit = {}, {}
    for key, name in [('normalised_crps', 'normalized_crps'), ('crps', 'crps'), ('mae', 'mae'), ('rmse', 'rmse')]:
        files = list(output.glob(f'evaluation/seed_m*_d*/{key}.npz'))
        if len(files) != 1:
            raise ValueError(f'Missing or ambiguous metric artifact: {key}')
        with np.load(files[0], allow_pickle=False) as array:
            values = array['loss']
            metrics[name], metric_audit[name] = summarize_author_metric(values, key)
    status.update(status='completed', metrics=metrics, evaluation_seconds=time.monotonic() - started,
                  metric_audit=metric_audit,
                  paper_score_status='single_run_requires_aggregation')
    if settings.get('dual_air36_cutoff'):
        corrected = {}
        for key, name in [('normalised_crps', 'normalized_crps'), ('crps', 'crps'), ('mae', 'mae'), ('rmse', 'rmse')]:
            files = list(output.glob(f'evaluation/seed_m*_d*/corrected_{key}.npz'))
            if len(files) != 1:
                raise ValueError('Missing corrected Air-36 evaluation')
            with np.load(files[0], allow_pickle=False) as payload:
                corrected[name], _ = summarize_author_metric(payload['loss'], key)
        status['corrected_metrics'] = corrected
        status['correction_boundary'] = 'Released CFMI evaluation trims features instead of time for overlap cutoff. Paper comparison retains released metric; corrected metric trims time and is separate.'
    if settings.get('metric_protocol'):
        shared = json.loads((output / 'shared_metrics.json').read_text(encoding='utf-8'))
        status['author_metrics'] = metrics
        status.update(shared)
        status['paper_score_status'] = 'transfer_requires_paired_aggregation'
    result_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config)), flush=True)
    except Exception as exc:
        settings = json.loads(args.config.read_text(encoding='utf-8'))
        output = ROOT / settings['output_root']
        output.mkdir(parents=True, exist_ok=True)
        (output / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}), encoding='utf-8')
        raise
