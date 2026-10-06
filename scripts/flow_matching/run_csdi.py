"""Run the author CSDI implementation on audited, config-selected arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.flow_matching.metrics import ensemble_totals, finalize, PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


class AuditedDataset(Dataset):
    def __init__(self, data, split):
        self.values = data['values']
        self.observed = data['observed']
        self.conditioning = data['conditioning']
        self.indices = data[split]
        self.historical = data.get('hist_mask')
        self.cut_length = data.get('cut_length')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        result = {'observed_data': self.values[index], 'observed_mask': self.observed[index].astype(np.float32),
                'gt_mask': self.conditioning[index].astype(np.float32), 'timepoints': np.arange(self.values.shape[1])}
        if self.historical is not None:
            result['hist_mask'] = self.historical[index].astype(np.float32)
            result['cut_length'] = self.cut_length[index]
        return result


def train_resumable(model, config, train_loader, val_loader, output, configuration_hash):
    """Author Adam/schedule/loss/final-checkpoint policy, with restart snapshots."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[int(.75 * config['epochs']), int(.9 * config['epochs'])], gamma=.1)
    resume = output / 'training_resume.pt'
    start, best = 0, float('inf')
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)  # Locally generated restart state only.
        if state['config_sha256'] != configuration_hash:
            raise ValueError('Restart state belongs to a different configuration')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        start, best = state['next_epoch'], state['best_valid']
    for epoch in range(start, config['epochs']):
        started, losses = time.monotonic(), []
        model.train()
        for batch_number, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = model(batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if batch_number >= config['itr_per_epoch']:
                break
        scheduler.step()
        validation = None
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                validation = sum(model(batch, is_train=0).item() for batch in val_loader)
            best = min(best, validation)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'next_epoch': epoch + 1, 'best_valid': best,
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                 'config_sha256': configuration_hash}
        temporary = resume.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, resume)
        progress = {'stage': 'training', 'epoch': epoch + 1, 'total_epochs': config['epochs'],
                    'loss': float(np.mean(losses)), 'validation_loss_sum': validation,
                    'epoch_seconds': time.monotonic() - started}
        (output / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')
        print(json.dumps(progress), flush=True)
    # The released CSDI train() evaluates the final epoch, not the best validation checkpoint.
    torch.save(model.state_dict(), output / 'model.pth')


def run(config_path):
    settings = json.loads(Path(config_path).read_text(encoding='utf-8'))
    output = ROOT / settings['output_root']
    output.mkdir(parents=True, exist_ok=True)
    finished = output / 'result.json'
    config_hash = file_hash(config_path)
    if finished.exists():
        record = json.loads(finished.read_text(encoding='utf-8'))
        if record['config_sha256'] != config_hash:
            raise ValueError('Existing run has another frozen configuration')
        return record
    source = ROOT / settings['author_source']
    sys.path.insert(0, str(source))
    from main_model import CSDI_Physio, CSDI_PM25
    from utils import calc_quantile_CRPS
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(settings.get('cpu_threads', 4))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load((source / settings.get('author_config', 'config/base.yaml')).read_text())
    config['model']['is_unconditional'] = False
    config['model']['test_missing_ratio'] = settings['missing_ratio']
    if settings.get('target_strategy'):
        config['model']['target_strategy'] = settings['target_strategy']
    config['train']['epochs'] = settings.get('epochs', config['train']['epochs'])
    data_path = ROOT / settings['processed_data']
    if settings.get('processed_sha256') and file_hash(data_path) != settings['processed_sha256']:
        raise ValueError('Frozen processed data changed')
    with np.load(data_path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(data[a]) & set(data[b]):
            raise ValueError('Group leakage')
    if not np.array_equal(data['eval_mask'], data['observed'] & ~data['conditioning']):
        raise ValueError('Invalid evaluation targets')
    model_class = CSDI_PM25 if settings.get('author_kind') == 'pm25' else CSDI_Physio
    model = model_class(config, settings['device'], target_dim=data['values'].shape[-1]).to(settings['device'])
    metadata = {'id': settings['id'], 'config_sha256': config_hash,
                'config': settings, 'processed_sha256': file_hash(data_path),
                'author_snapshot': json.loads((source.parent / 'snapshot.json').read_text()),
                'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
                'status': 'running', 'pid': os.getpid(), 'metrics': {}, 'eval_count': 0,
                'patch_manifest': json.loads((source.parent / 'patch_manifest.json').read_text()),
                'paper_score_status': 'pending_multi_run_protocol_check'}
    (output / 'status.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    checkpoint = ROOT / settings['checkpoint'] if settings.get('checkpoint') else output / 'model.pth'
    started = time.monotonic()
    if settings['mode'] == 'train_evaluate' and not checkpoint.exists():
        train_loader = DataLoader(AuditedDataset(data, 'train'), batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(AuditedDataset(data, 'val'), batch_size=config['train']['batch_size'], shuffle=False)
        train_resumable(model, config['train'], train_loader, val_loader, output, config_hash)
        checkpoint = output / 'model.pth'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=settings['device'], weights_only=True))
    metadata['checkpoint_sha256'] = file_hash(checkpoint)
    metadata['training_seconds'] = time.monotonic() - started
    model.eval()
    test = AuditedDataset(data, 'test')
    if settings.get('limit_test_cases'):
        test.indices = test.indices[:settings['limit_test_cases']]
    loader = DataLoader(test, batch_size=settings.get('evaluation_batch_size', 16), shuffle=False)
    totals = {'mae': 0., 'mse': 0., 'crps_numerator': 0., 'crps_denominator': 0., 'eval_count': 0}
    shared_totals = {}
    started = time.monotonic()
    with torch.inference_mode():
        for number, batch in enumerate(loader, 1):
            samples, target, mask, _, _ = model.evaluate(batch, settings['num_samples'])
            samples = samples.permute(0, 1, 3, 2)
            target = target.permute(0, 2, 1)
            mask = mask.permute(0, 2, 1)
            native_target, native_samples = target, samples
            if settings.get('metric_scale') == 'physical':
                scale = torch.as_tensor(data['std'], device=target.device, dtype=target.dtype)
                mean = torch.as_tensor(data['mean'], device=target.device, dtype=target.dtype)
                target = target * scale + mean
                samples = samples * scale + mean
            if settings.get('metric_protocol') == PROTOCOL:
                addition = ensemble_totals(target.cpu().numpy(), samples.cpu().numpy(), mask.cpu().numpy())
                for key, value in addition.items():
                    shared_totals[key] = shared_totals.get(key, 0) + value
            point = samples.median(dim=1).values
            error = (native_samples.median(dim=1).values - native_target) * scale if settings.get('metric_scale') == 'physical' else point - target
            count = mask.sum().item()
            totals['mae'] += (torch.abs(error) * mask).sum().item()
            totals['mse'] += ((error ** 2) * mask).sum().item()
            denominator = torch.abs(target * mask).sum().item()
            batch_crps = calc_quantile_CRPS(target, samples, mask, 0, 1)
            totals['crps_numerator'] += batch_crps * denominator
            totals['crps_denominator'] += denominator
            totals['eval_count'] += count
            progress = {'batch': number, 'total_batches': len(loader), 'elapsed_seconds': time.monotonic() - started,
                        'mae': totals['mae'] / totals['eval_count'], 'rmse': (totals['mse'] / totals['eval_count']) ** .5}
            (output / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')
            print(json.dumps(progress), flush=True)
    if not totals['eval_count']:
        raise ValueError('No evaluable held-out observations')
    metadata.update(status='completed', evaluation_seconds=time.monotonic() - started,
        test_cases=len(test), eval_count=totals['eval_count'],
        metrics={'mae': totals['mae'] / totals['eval_count'], 'rmse': (totals['mse'] / totals['eval_count']) ** .5,
                 'normalized_crps': totals['crps_numerator'] / totals['crps_denominator']},
        paper_score_status='diagnostic_only' if settings.get('diagnostic') else 'single_run_requires_aggregation')
    if settings.get('metric_protocol') == PROTOCOL:
        metadata['author_metrics'] = metadata['metrics']
        metadata['metrics'] = finalize(shared_totals)
        metadata['metric_protocol'] = PROTOCOL
        metadata['paper_score_status'] = 'transfer_requires_paired_aggregation'
    finished.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    (output / 'status.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config)), flush=True)
    except Exception as exc:
        settings = json.loads(args.config.read_text(encoding='utf-8'))
        directory = ROOT / settings['output_root']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}), encoding='utf-8')
        raise
