"""The paper's explicit50-epoch exception, full native image data and backbone."""
import math
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT, read, sha, validate_history, verify_receipts,
)


def validate_tccm_job(job, model, optimizer, dataset):
    expected = dict(epochs=50, batch_size=32, gradient_accumulation=1,
        mixed_precision='bfloat16', TF32=True, early_stopping=False, scheduler=None,
        EMA=False, drop_last=False, shuffle=True, distributed_data_parallel=False,
        world_size=1, cpu_threads=2, checkpoint_interval=10)
    if job['training'] != expected or job['method'] != 'TCCM_controlled_image_domain':
        raise ValueError('Original50-epoch TCCM exception or frozen local choices differ')
    if (job['dataset'] != 'mvtec' or job['variant'] != 'conditional'
            or job['seed'] not in range(1103, 1108)
            or dataset['expected_native_splits'] != {'train':3629, 'test':1725}
            or dataset['expected_original_category_count'] != 15
            or not dataset['complete_raw_data_verified'] or not dataset['complete_derived_data_verified']):
        raise ValueError('Complete original MVTec controlled image comparison required')
    parameters = model['parameters']
    architecture = dict(num_classes=15, model_channels=64, num_res_blocks=3,
        channel_mult=[1,1,2,2,4,4], attention_resolutions=[4,8,16,32],
        num_head_channels=32, dropout=.05)
    if any(parameters.get(k) != v for k, v in architecture.items()):
        raise ValueError('Full same-backbone controlled TCCM architecture differs')
    opt = optimizer['parameters']
    if any(opt.get(k) != v for k,v in dict(learning_rate=1e-4, betas=[.9,.99],
            epsilon=1e-6, weight_decay=1e-4, amsgrad=True, update_rms_clip_threshold=1.).items()):
        raise ValueError('Identical controlled-comparison optimizer settings required')
    if (job['expected_training_images_per_epoch'] != 3629
            or job['expected_optimizer_steps'] != 50 * math.ceil(3629/32)):
        raise ValueError('All full original normal images and50-epoch update budget required')
    return 3629


def completion_valid(job, result, root=ROOT):
    if (result.get('job_id') != job['id'] or result.get('seed') != job['seed']
            or result.get('completed_epochs') != 50 or result.get('global_steps') != 5700
            or result.get('benchmark_performance') is not False
            or result.get('original_author_equivalence_certified') is not False):
        raise ValueError('Full original50-epoch TCCM completion only')
    validate_history(result['history'],50,3629)
    if len(result['epoch_receipts']) != 50 or len({x['path'] for x in result['epoch_receipts']}) != 50:
        raise ValueError('Every full TCCM epoch receipt required')
    if [Path(x['path']).name for x in result['checkpoint_receipts']] != [
            'epoch_%04d.pt' % epoch for epoch in range(10,51,10)]:
        raise ValueError('All preregistered full TCCM checkpoints required')
    config_path = root/result['job_configuration']
    if read(config_path) != job or result['binding']['job_sha256'] != sha(config_path):
        raise ValueError('Original full TCCM job configuration binding differs')
    for kind in ('dataset','model','optimizer'):
        if result['binding'][kind+'_configuration_sha256'] != sha(root/job[kind+'_configuration']):
            raise ValueError('Full native TCCM source/data binding differs')
    verify_receipts(result['checkpoint_receipts'],root); verify_receipts(result['epoch_receipts'],root)
    for record,receipt in zip(result['history'],result['epoch_receipts']):
        path = Path(receipt['path']); path = path if path.is_absolute() else root/path
        if read(path) != record:
            raise ValueError('Full epoch ledger differs from immutable original training receipt')
    return True
