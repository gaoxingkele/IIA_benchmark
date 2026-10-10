"""Audit complete released LS4 climate loaders on both preserved split branches."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import math
import os
import sys

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import read, sha, write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned full climate loader input changed: '+item['path'])


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    verify(config)
    lock = read(ROOT / config['environment_lock'])
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k,v in lock['packages'].items()):
        raise ValueError('Original locked LS4 loader environment required')
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / config['source_root']))
    import datasets
    torch.set_num_threads(config['cpu_threads'])
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every previous loader audit')
    target.mkdir(parents=True)
    frame = pd.read_csv(ROOT / config['csv_path']).astype(np.float32)
    all_ids = frame.ID.unique().astype(np.int64)
    before = set(frame.loc[frame.Time <= 150, 'ID'])
    after = set(frame.loc[frame.Time > 150, 'ID'])
    eligible = before & after
    columns = [c for c in frame.columns if c != 'ID']
    value_columns = [f'Value_{i}' for i in range(5)]
    mask_columns = [f'Mask_{i}' for i in range(5)]
    groups = {int(i):g.sort_values('Time') for i,g in frame.groupby('ID')}
    profiles, batches = [], []
    for branch in ('source_literal', 'id_corrected'):
        for fold in range(5):
            split_dir = ROOT / config['prepared_directory'] / branch / f'small_chunk_fold_idx_{fold}'
            indices = {k:np.load(split_dir / f'{k}_idx.npy',allow_pickle=False) for k in ('train','val','test')}
            original_ids = {k:np.asarray([i for i in all_ids if i in set(v) and (k=='train' or i in eligible)]) for k,v in indices.items()}
            selected = pd.concat([frame[frame.ID.isin(original_ids['train'])],frame[frame.ID.isin(original_ids['val']) & (frame.Time<=150)]])
            independent_min = np.array([selected.loc[selected[m] == 1,v].min() for v,m in zip(value_columns,mask_columns)], dtype=np.float32)
            independent_max = np.array([selected.loc[selected[m] == 1,v].max() for v,m in zip(value_columns,mask_columns)], dtype=np.float32)
            for source_config in config['native_configs']:
                yaml = OmegaConf.load(ROOT / source_config)
                arguments = OmegaConf.create(OmegaConf.to_container(yaml.data,resolve=True))
                arguments.base = str((ROOT / config['csv_path']).parent.resolve())
                arguments.dir = str(split_dir.resolve())
                batch_size = int(yaml.optim.batch_size)
                if batch_size != 64 or yaml.optim.epochs != 500 or arguments.channel != 5 or arguments.T_val != 150 or arguments.max_val_samples != 3:
                    raise ValueError('Original scientific climate YAML changed')
                objects = datasets.parse_datasets(arguments,batch_size=batch_size,device=torch.device('cpu'))
                for label, native_set, ids in (
                    ('train',objects['train_dataloader'].dataset.datasets[0],original_ids['train']),
                    ('val',objects['train_dataloader'].dataset.datasets[1],original_ids['val']),
                    ('test',objects['test_dataloader'].dataset,original_ids['test'])):
                    if len(native_set) != len(ids):
                        raise ValueError('Full actual source series count differs')
                    for position, raw_id in enumerate(ids):
                        actual = native_set[position]['path'][columns].to_numpy()
                        expected = groups[int(raw_id)]
                        if label != 'train':
                            expected = expected[expected.Time<=150]
                        if not np.array_equal(actual, expected[columns].to_numpy()):
                            raise ValueError('Complete native raw values/time/mask differ')
                closure = dict(zip(objects['train_dataloader'].collate_fn.__code__.co_freevars,
                                   [c.cell_contents for c in objects['train_dataloader'].collate_fn.__closure__]))
                if not np.array_equal(closure['data_min'].numpy(),independent_min) or not np.array_equal(closure['data_max'].numpy(),independent_max):
                    raise ValueError('Source train-plus-validation-prefix normalizer differs')
                counts = dict(train=len(original_ids['train'])+len(original_ids['val']),test=len(original_ids['test']))
                record = dict(branch=branch,fold=fold,source_config=source_config,extrapolation=bool(arguments.extrap),
                              grouped_series_counts={k:len(v) for k,v in original_ids.items()},
                              complete_source_masks_values_and_times_verified=True,
                              normalization_min=independent_min.tolist(),normalization_max=independent_max.tolist(),
                              full_train_batches=math.ceil(counts['train']/batch_size),full_test_batches=math.ceil(counts['test']/batch_size),
                              expected_full_training_optimizer_updates=500*math.ceil(counts['train']/batch_size),
                              source_test_future_samples_not_used_by_collate=True,
                              source_test_future_samples_available=len(objects['test_dataloader'].dataset.df_after))
                for split in ('train','test'):
                    loader = objects[split+'_dataloader']
                    covered = 0
                    split_batches = 0
                    for index,data in enumerate(loader):
                        count = len(data['observed_data'])
                        if count != min(batch_size, counts[split]-covered):
                            raise ValueError('Complete original batch coverage differs')
                        for key in ('observed_data','observed_tp','observed_mask','data_to_predict','tp_to_predict','mask_predicted_data'):
                            if not torch.isfinite(data[key]).all():
                                raise ValueError('Nonfinite native original loader tensors')
                        if data['observed_data'].shape[-1] != 5 or data['data_to_predict'].shape[-1] != 5:
                            raise ValueError('All5 variables required')
                        if not arguments.extrap and not torch.equal(data['observed_data'],data['data_to_predict']):
                            raise ValueError('Original interpolation reconstructs identical observed targets')
                        batches.append(dict(branch=branch,fold=fold,source_config=source_config,split=split,batch_index=index,
                            series=count,observed_shape=list(data['observed_data'].shape),predicted_shape=list(data['data_to_predict'].shape),
                            observed_mask_count=int(data['observed_mask'].sum()),predicted_mask_count=int(data['mask_predicted_data'].sum())))
                        covered += count
                        split_batches += 1
                        write(target/'progress.json',dict(branch=branch,fold=fold,source_config=source_config,split=split,
                            covered_series=covered,required_series=counts[split],completed_profiles=len(profiles),required_profiles=20))
                    if covered != counts[split] or split_batches != math.ceil(counts[split]/batch_size):
                        raise ValueError('Every original batch and series required')
                profiles.append(record)
                print(branch,fold,source_config,counts,flush=True)
                del objects,loader,closure,native_set
                gc.collect()
    verify(config)
    if len(profiles) != 20 or torch.cuda.is_initialized():
        raise ValueError('Both branches/all5folds/both full original CPU profiles required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=config_path,
                  configuration_sha256=sha(ROOT/config_path),full_original_loader_passed=True,
                  full_profiles=len(profiles),full_batches=len(batches),profiles=profiles,batches=batches,
                  historical_author_split_equivalence_certified=False,performance_result=False,
                  source_test_future_samples_not_used_by_collate=True,interpolation_observed_target_overlap=True,
                  raw_csv_preserved=True,all_original_experiments_complete=False,boundary=config['boundary'])
    write(target/'original_loader_audit.json',report)
    write(target/'validation.json',dict(source_receipts=config['source_receipts'],outputs={
        'original_loader_audit.json':sha(target/'original_loader_audit.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    worker(read(ROOT/args.config),args.config)


if __name__ == '__main__':
    main()
