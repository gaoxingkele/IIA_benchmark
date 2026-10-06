"""Exercise the pinned GRIN training/validation/test CLI on real Air-36 data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import numpy as np
    import torch
    import pytorch_lightning as pl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_grin_cpu_preflight.v1.json')
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding='utf-8'))
    if configuration.get('diagnostic') is not True:
        raise ValueError('This runner only supports diagnostic configurations')
    source = ROOT / configuration['author_source']
    sys.path.insert(0, str(source))
    import lib
    data_config = json.loads((ROOT / configuration['data_config']).read_text(encoding='utf-8'))
    data_root = ROOT / data_config['output_root']
    audit = json.loads((data_root / 'data_audit.json').read_text(encoding='utf-8'))
    data_sha = sha(data_root / 'small36.h5')
    if data_sha != audit['files']['small36.h5']['sha256']:
        raise ValueError('Staged GRIN data changed')
    output = ROOT / configuration['output_root']
    output.mkdir(parents=True, exist_ok=True)
    lib.datasets_path['air'] = str(data_root)
    lib.config['logs'] = str(output / 'author_logs')
    spec = importlib.util.spec_from_file_location('grin_author_cli', source / 'scripts/run_imputation.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    with patch.object(sys, 'argv', ['run_imputation.py', '--model-name', 'grin', '--config', str(ROOT / configuration['author_config'])]):
        args = author.parse_args()
    # YAML is applied after CLI parsing by the original author; override diagnostics afterward.
    for key in ('seed', 'epochs', 'batch_size', 'samples_per_epoch'):
        setattr(args, key, configuration[key])
    args.workers = 0
    args.dataset_name = 'air36'
    capture = {'metrics': {}, 'original_split_windows': {}}
    get_dataset = author.get_dataset

    def diagnostic_dataset(name):
        dataset = get_dataset(name)
        original_splitter = dataset.splitter

        def splitter(torch_dataset, val_len=1., in_sample=False, window=0):
            splits = original_splitter(torch_dataset, val_len, in_sample, window)
            selected = []
            for label, indices in zip(('train', 'validation', 'test'), splits):
                capture['original_split_windows'][label] = len(indices)
                indices = np.asarray(indices)
                good = []
                for index in indices:
                    expanded = torch_dataset.expand_indices([int(index)], merge=True)
                    if dataset.eval_mask[expanded].sum() > 0:
                        good.append(int(index))
                    if len(good) == configuration['subset_windows']:
                        break
                if len(good) != configuration['subset_windows']:
                    raise ValueError('Not enough real masked windows in ' + label)
                selected.append(np.asarray(good))
            return selected

        dataset.splitter = splitter
        return dataset

    author.get_dataset = diagnostic_dataset
    real_setup = author.SpatioTemporalDataModule.setup

    def setup(dm, stage=None):
        result = real_setup(dm, stage)
        capture['dm'] = dm
        return result

    for label in ('mae', 'mse', 'mre', 'mape'):
        metric = getattr(author.numpy_metrics, 'masked_' + label)

        def metric_wrapper(prediction, target, mask, _metric=metric, _label=label):
            result = _metric(prediction, target, mask)
            if not np.isfinite(result):
                raise ValueError('Non-finite GRIN test ' + _label)
            capture['metrics'][_label] = float(result)
            capture['test_targets'] = int(mask.sum())
            return result

        setattr(author.numpy_metrics, 'masked_' + label, metric_wrapper)
    # Preserve the model, optimizer, scheduler, validation checkpoint selection and metrics.
    with patch.object(author.SpatioTemporalDataModule, 'setup', setup), patch.object(torch.cuda, 'is_available', return_value=False):
        truth, prediction, mask = author.run_experiment(args)
    if not np.isfinite(prediction).all() or set(capture['metrics']) != {'mae', 'mse', 'mre', 'mape'}:
        raise ValueError('Incomplete GRIN author evaluation')
    checkpoints = list((output / 'author_logs').rglob('*.ckpt'))
    if not checkpoints or capture['test_targets'] <= 0:
        raise ValueError('No validation checkpoint or test targets')
    checkpoint = torch.load(str(checkpoints[-1]), map_location='cpu')
    dm = capture.pop('dm')
    result = {'status': 'passed', 'diagnostic': True, 'config_sha256': sha(arguments.config),
              'author_config_sha256': sha(ROOT / configuration['author_config']), 'data_sha256': data_sha,
              'versions': {'python': sys.version.split()[0], 'torch': torch.__version__, 'lightning': pl.__version__},
              'model_state_elements_including_buffers': sum(value.numel() for key, value in checkpoint['state_dict'].items() if key.startswith('model.')),
              'checkpoint_epoch': int(checkpoint['epoch']), 'validation_checkpoint': str(checkpoints[-1].relative_to(ROOT)),
              'split_windows': {label: len(subset) for label, subset in zip(('train', 'validation', 'test'), (dm.trainset, dm.valset, dm.testset))},
              'prediction_shape': list(prediction.shape), 'finite_predictions': True,
              'boundary': configuration['boundary'], **capture}
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
