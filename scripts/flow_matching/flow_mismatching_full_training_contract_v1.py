"""Frozen full-image training bindings; imports no model or array libraries."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_receipts(receipts, root=ROOT):
    for item in receipts:
        path = Path(item['path'])
        if not path.is_absolute():
            path = root / path
        if sha(path) != item['sha256']:
            raise ValueError('Frozen source or data differs: ' + str(path))


def validate_job(job, model, optimizer, dataset):
    expected = dict(epochs=2000, batch_size=32, gradient_accumulation=1,
                    mixed_precision='bfloat16', TF32=True, early_stopping=False,
                    scheduler=None, EMA=False, drop_last=False, shuffle=True,
                    distributed_data_parallel=False, world_size=1, cpu_threads=2,
                    checkpoint_interval=100)
    if job['training'] != expected:
        raise ValueError('Full budget or frozen local training choice differs')
    variant = job['variant']
    if variant not in ('conditional', 'pooled'):
        raise ValueError('Unknown conditioning ablation')
    count = dataset['expected_original_category_count']
    parameters = model['parameters']
    architecture = dict(model_channels=64, num_res_blocks=3,
                        channel_mult=[1, 1, 2, 2, 4, 4],
                        attention_resolutions=[4, 8, 16, 32],
                        num_head_channels=32, dropout=0.05)
    if any(parameters.get(k) != v for k, v in architecture.items()):
        raise ValueError('Original full U-Net architecture differs')
    if parameters['num_classes'] != (count if variant == 'conditional' else None):
        raise ValueError('Conditioning does not match registered ablation')
    if optimizer['parameters'] != dict(
            source_path=optimizer['parameters']['source_path'],
            expected_sha256=optimizer['parameters']['expected_sha256'],
            learning_rate=1e-4, betas=[0.9, 0.99], epsilon=1e-6,
            weight_decay=1e-4, amsgrad=True, update_rms_clip_threshold=1.0):
        raise ValueError('Original optimizer hyperparameters differ')
    if not dataset['complete_raw_data_verified'] or not dataset['complete_derived_data_verified']:
        raise ValueError('Full independently verified dataset required')
    n = dataset['expected_native_splits']['train']
    if job['expected_training_images_per_epoch'] != n:
        raise ValueError('Subset training is forbidden')
    if job['expected_optimizer_steps'] != 2000 * math.ceil(n / 32):
        raise ValueError('Complete optimizer update budget differs')
    if job['seed'] not in range(1103, 1108):
        raise ValueError('Seed outside preregistered local uncertainty cohort')
    return n


def validate_history(history, epochs, n, batch_size=32):
    if len(history) != epochs:
        raise ValueError('Incomplete epoch coverage ledger')
    for epoch, item in enumerate(history, 1):
        if (item['epoch'] != epoch or item['images_visited'] != n
                or item['unique_images_visited'] != n
                or item['optimizer_steps'] != math.ceil(n / batch_size)
                or not math.isfinite(item['mean_training_loss'])
                or len(item['permutation_sha256']) != 64):
            raise ValueError('Full epoch coverage or finite loss differs')


def completion_valid(job, result, root=ROOT):
    if (result.get('job_id') != job['id'] or result.get('seed') != job['seed']
            or result.get('completed_epochs') != 2000
            or result.get('global_steps') != job['expected_optimizer_steps']
            or result.get('benchmark_performance') is not False
            or result.get('original_author_equivalence_certified') is not False):
        raise ValueError('Incomplete or mislabelled full training result')
    validate_history(result['history'], 2000, job['expected_training_images_per_epoch'])
    config_path = root / result['job_configuration']
    if read(config_path) != job or result['binding']['job_sha256'] != sha(config_path):
        raise ValueError('Complete result job binding differs')
    for kind in ('dataset', 'model', 'optimizer'):
        path = job[kind + '_configuration']
        if result['binding'][kind + '_configuration_sha256'] != sha(root / path):
            raise ValueError('Complete result source binding differs')
    if len(result['checkpoint_receipts']) != 20 or len(result['epoch_receipts']) != 2000:
        raise ValueError('Complete periodic checkpoint and epoch receipts required')
    if len({x['path'] for x in result['epoch_receipts']}) != 2000:
        raise ValueError('Distinct full epoch receipts required')
    if [Path(x['path']).name for x in result['checkpoint_receipts']] != [
            'epoch_%04d.pt' % x for x in range(100, 2001, 100)]:
        raise ValueError('Complete original epoch checkpoint grid differs')
    for record, receipt in zip(result['history'], result['epoch_receipts']):
        path = Path(receipt['path'])
        if not path.is_absolute():
            path = root / path
        if read(path) != record:
            raise ValueError('Checkpoint history and immutable epoch record differ')
    verify_receipts(result['checkpoint_receipts'], root)
    verify_receipts(result['epoch_receipts'], root)
    return True
