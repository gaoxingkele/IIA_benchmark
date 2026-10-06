"""Run the frozen GRIN author CLI, preserving its original loss and point metrics."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpochRNG(pl.Callback):
    """Persist sampler/masking randomness alongside author optimizer/callback state."""
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.restored = callback_state

    def on_train_start(self, trainer, pl_module):
        # Lightning may seed again while setting up loaders; restore immediately before training.
        if hasattr(self, 'restored'):
            state = self.restored
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'])
            if state['cuda'] is not None:
                torch.cuda.set_rng_state_all(state['cuda'])

    def on_train_end(self, trainer, pl_module):
        self.completed_epochs = int(trainer.current_epoch) + 1


def month_statistics(prediction, target, mask, months):
    error = prediction - target
    records = []
    for month in np.unique(months):
        selected = (months == month)[:, None] & mask.astype(bool)
        records.append([float(np.abs(error)[selected].sum()), float(np.square(error)[selected].sum()),
                        int(selected.sum()), float(target[selected].sum())])
    return np.unique(months), np.asarray(records)


def bootstrap(records, seed, draws=2000):
    generator = np.random.default_rng(seed)
    totals = records[generator.integers(0, len(records), size=(draws, len(records)))].sum(axis=1)
    valid = (totals[:, 2] > 0) & (totals[:, 3] > 0)
    totals = totals[valid]
    if not len(totals):
        raise ValueError('No evaluable month bootstrap draws')
    values = np.stack((totals[:, 0] / totals[:, 2], np.sqrt(totals[:, 1] / totals[:, 2]),
                       totals[:, 0] / (totals[:, 3] + 1e-8)), axis=1)
    return {key: np.quantile(values[:, index], [.025, .975]).tolist()
            for index, key in enumerate(('mae', 'rmse', 'mre'))}


def run(config_path, stop_after_epoch=None):
    started = time.time()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    config_sha = sha(config_path)
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=True)
    identity = output / 'run_identity.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))['config_sha256'] != config_sha:
        raise ValueError('Checkpoint configuration differs from frozen job')
    identity.write_text(json.dumps({'config_sha256': config_sha}), encoding='utf-8')
    diagnostic = bool(cfg.get('diagnostic'))
    report_path = output / ('diagnostic_report.json' if diagnostic else 'result.json')
    if report_path.exists():
        result = json.loads(report_path.read_text(encoding='utf-8'))
        if result['config_sha256'] != config_sha:
            raise ValueError('Result configuration mismatch')
        return result
    if stop_after_epoch and not diagnostic:
        raise ValueError('Deliberate interruption is restricted to diagnostics')
    author_config = ROOT / cfg['author_config']
    if sha(author_config) != cfg['author_config_sha256']:
        raise ValueError('Author YAML hyperparameters changed')
    source = ROOT / cfg['author_source']
    snapshot = json.loads((source.parent / 'snapshot.json').read_text(encoding='utf-8'))
    if snapshot['commit'] != cfg['author_commit']:
        raise ValueError('Author commit differs from registration')
    data_config = json.loads((ROOT / cfg['data_config']).read_text(encoding='utf-8'))
    data_root = ROOT / data_config['output_root']
    filename = 'small36.h5' if cfg['dataset'] == 'air36' else 'full437.h5'
    if sha(data_root / filename) != cfg['data_sha256']:
        raise ValueError('Registered raw graph dataset changed')
    sys.path.insert(0, str(source))
    import lib
    lib.datasets_path['air'] = str(data_root)
    lib.config['logs'] = str(output / 'author_logs')
    spec = importlib.util.spec_from_file_location('grin_author_cli', source / 'scripts/run_imputation.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    with patch.object(sys, 'argv', ['run_imputation.py', '--model-name', 'grin', '--config', str(author_config)]):
        args = author.parse_args()
    args.seed = cfg['seed']
    if args.dataset_name != cfg['dataset'] or args.model_name != 'grin':
        raise ValueError('Author model/dataset do not match configuration')
    if diagnostic:
        args.epochs = cfg['diagnostic_epochs']
        args.batch_size = 2
        args.samples_per_epoch = 4
    args.workers = 0
    capture = {'metrics': {}}
    get_dataset = author.get_dataset

    def dataset_loader(name):
        dataset = get_dataset(name)
        splitter = dataset.splitter

        def audited_split(torch_dataset, val_len=1., in_sample=False, window=0):
            splits = splitter(torch_dataset, val_len, in_sample, window)
            full_slices = [torch_dataset.expand_indices(indices, merge=True) for indices in splits]
            names = ('train', 'validation', 'test')
            audit = {'window_counts': dict(zip(names, map(len, splits))),
                     'unique_timestamp_counts': dict(zip(names, map(len, full_slices))),
                     'timestamp_overlap_counts': {names[a] + '_' + names[b]: int(len(np.intersect1d(full_slices[a], full_slices[b])))
                         for a, b in ((0, 1), (0, 2), (1, 2))},
                     'original_eval_targets': int(dataset.eval_mask.sum()),
                     'eval_targets_unknown': int((dataset.eval_mask.astype(bool) & ~dataset.mask.astype(bool)).sum()),
                     'normalization': 'Author train timestamps only, excluding evaluation coordinates before restoring training evaluation values.',
                     'native_nan_fill': 'Original weekly/hour mean over released full series; original observed/evaluation masks remain enforced.'}
            if audit['eval_targets_unknown']:
                raise ValueError('Evaluation targets include original missing values')
            capture['data_audit'] = audit
            if diagnostic:
                splits = [np.asarray(indices[:2]) for indices in splits]
            return splits

        dataset.splitter = audited_split
        return dataset

    real_setup = author.SpatioTemporalDataModule.setup

    def setup(dm, stage=None):
        value = real_setup(dm, stage)
        capture['dm'] = dm
        return value

    for label in ('mae', 'mse', 'mre', 'mape'):
        metric = getattr(author.numpy_metrics, 'masked_' + label)

        def measured(prediction, target, mask, _metric=metric, _label=label):
            value = _metric(prediction, target, mask)
            if not np.isfinite(value):
                raise ValueError('Non-finite whole-series ' + _label)
            capture['metrics'][_label] = float(value)
            if _label == 'mae':
                dm = capture['dm']
                # Identical chronology to author's df_true = dataset.df.iloc[dm.test_slice].
                time_index = dm.torch_dataset.index[dm.test_slice]
                months = time_index.strftime('%Y-%m').to_numpy()
                groups, records = month_statistics(prediction, target, mask, months)
                capture['group_ids'], capture['group_statistics'] = groups, records
                capture['test_targets'] = int(mask.sum())
            return value

        setattr(author.numpy_metrics, 'masked_' + label, measured)
    checkpoint_class = author.ModelCheckpoint
    checkpoint_directory = output / 'checkpoints'

    def checkpoint_factory(**kwargs):
        kwargs.update(dirpath=str(checkpoint_directory), save_last=True)
        callback = checkpoint_class(**kwargs)
        capture['checkpoint_callback'] = callback
        return callback

    trainer_class = author.pl.Trainer

    class AuditedTrainer(trainer_class):
        def __init__(self, **kwargs):
            rng_callback = EpochRNG()
            kwargs['callbacks'].append(rng_callback)
            capture['rng_callback'] = rng_callback
            kwargs['progress_bar_refresh_rate'] = 0
            if (checkpoint_directory / 'last.ckpt').exists():
                kwargs['resume_from_checkpoint'] = str(checkpoint_directory / 'last.ckpt')
            if stop_after_epoch:
                kwargs['max_epochs'] = stop_after_epoch
            super().__init__(**kwargs)
            capture['trainer'] = self

    if cfg['device'] == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('Frozen GPU job requires CUDA; CPU fallback would change execution')
    if cfg['device'] not in ('cpu', 'cuda:0'):
        raise ValueError('Only CPU and cuda:0 are registered')
    with ExitStack() as stack:
        if cfg['device'] == 'cpu':
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        stack.enter_context(patch.object(author.SpatioTemporalDataModule, 'setup', setup))
        stack.enter_context(patch.object(author, 'get_dataset', dataset_loader))
        stack.enter_context(patch.object(author, 'ModelCheckpoint', checkpoint_factory))
        stack.enter_context(patch.object(author.pl, 'Trainer', AuditedTrainer))
        author.run_experiment(args)
    if stop_after_epoch:
        return {'status': 'interrupted_for_diagnostic_resume', 'config_sha256': config_sha}
    if set(capture['metrics']) != {'mae', 'mse', 'mre', 'mape'}:
        raise ValueError('Incomplete author metric aggregation')
    dm, trainer, callback = (capture.pop(key) for key in ('dm', 'trainer', 'checkpoint_callback'))
    groups, records = capture.pop('group_ids'), capture.pop('group_statistics')
    np.savez_compressed(output / 'test_month_statistics.npz', group_ids=groups, statistics=records)
    capture['metrics']['rmse'] = float(np.sqrt(capture['metrics']['mse']))
    result = {'status': 'completed', 'config_sha256': config_sha, 'config': cfg,
              'diagnostic': diagnostic, 'author_snapshot': snapshot, 'author_arguments': vars(args),
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'lightning': pl.__version__},
              'completed_epochs': capture.pop('rng_callback').completed_epochs, 'selected_checkpoint': str(Path(callback.best_model_path).relative_to(ROOT)),
              'parameters': sum(parameter.numel() for parameter in trainer.lightning_module.model.parameters()),
              'month_block_bootstrap_95ci': bootstrap(records, cfg['seed']), 'group_count': len(groups),
              'uncertainty_boundary': 'Resample whole test calendar months after original mean aggregation of overlapping window predictions. Few months limit interval precision; seed-level uncertainty requires all five runs.',
              'metric_boundary': 'Original masked MAE/MSE/MRE/MAPE in physical units and ratio scale; RMSE added as sqrt(MSE). No probabilistic score for this deterministic model.',
              'elapsed_seconds': time.time() - started, **capture}
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    os.replace(str(temporary), str(report_path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stop_after_epoch', type=int)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.stop_after_epoch)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
