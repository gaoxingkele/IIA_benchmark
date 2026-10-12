"""Preserved full-data views and untouched SHCL native loader audit, not scores."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil

from scripts.flow_matching.audit_shcl_released_operators_v1 import native, environment
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def data_environment():
    return dict(environment(), data_packages={name:importlib.metadata.version(name)
        for name in ('pandas','scikit-learn')})


def verify(config):
    for pin in config['source_receipts']:
        if native.sha(native.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen native data preparation changed: ' + pin['path'])
    model = native.read(native.ROOT / config['model_config']); native.verify(model)
    for dataset in config['datasets']:
        for pin in dataset['inputs'].values():
            if native.sha(native.ROOT / pin['path']) != pin['sha256']:
                raise ValueError('Preserved raw dataset changed: ' + pin['path'])
    reference = native.read(native.ROOT / config['reference_resource_runtime'])
    if (config['settings'] != reference['settings'] or config['resource_lock'] != reference['resource_lock']
            or config['environment'] != data_environment() or len(config['profiles']) != 13
            or {d['dataset'] for d in config['datasets']} != {'MSL','SMAP','PSM','SWaT','machine-3-11'}):
        raise ValueError('All five registered datasets, full13 profiles, actual packages and existing caps required')
    return model


def drop_timestamp(source, destination):
    """Preserve every feature token and row, dropping only the declared first column."""
    with Path(source).open('r',encoding='utf-8-sig',newline='') as incoming, Path(destination).open(
            'x',encoding='utf-8',newline='') as outgoing:
        reader=csv.reader(incoming); writer=csv.writer(outgoing)
        header=next(reader)
        if header[0]!='timestamp_(min)' or len(header)!=26:
            raise ValueError('Declared PSM timestamp plus25 feature columns required')
        writer.writerow(header[1:])
        for row in reader:
            if len(row)!=26: raise ValueError('Every original PSM feature row must be preserved')
            writer.writerow(row[1:])


def prepare_view(config, binding):
    import numpy as np
    import pandas as pd
    root=Path(config['view_root']); target=root/binding['dataset']; target.mkdir(parents=True,exist_ok=False)
    inputs={k:native.ROOT/v['path'] for k,v in binding['inputs'].items()}
    if binding['format']=='author_npy':
        for key, path in inputs.items(): shutil.copyfile(path,target/binding['output_filenames'][key])
        train=np.load(inputs['train'],allow_pickle=False); test=np.load(inputs['test'],allow_pickle=False)
        labels=np.load(inputs['label'],allow_pickle=False)
    elif binding['format']=='PSM_timestamp_csv':
        for key in ('train','test'): drop_timestamp(inputs[key],target/binding['output_filenames'][key])
        shutil.copyfile(inputs['label'],target/binding['output_filenames']['label'])
        train=pd.read_csv(inputs['train']).values[:,1:]; test=pd.read_csv(inputs['test']).values[:,1:]
        labels=pd.read_csv(inputs['label']).values[:,1:]
        for key, expected in [('train',train),('test',test)]:
            if not np.array_equal(pd.read_csv(target/binding['output_filenames'][key]).values,expected,equal_nan=True):
                raise ValueError('Full PSM feature-token transformation changed numeric values or row order')
    elif binding['format']=='SWAT_numeric_label_csv':
        frame_train=pd.read_csv(inputs['train']); frame_test=pd.read_csv(inputs['test'])
        if frame_train.columns[-1]!='Normal/Attack' or frame_test.columns.tolist()!=frame_train.columns.tolist():
            raise ValueError('Full SWaT feature schema and original trailing labels required')
        if not np.all(frame_train.values[:,-1]==0): raise ValueError('SWaT declared normal training labels required')
        train=frame_train.values[:,:-1].copy(); test=frame_test.values[:,:-1].copy(); labels=frame_test.values[:,-1].copy()
        for key,array in [('train',train),('test',test),('label',labels)]:
            with (target/binding['output_filenames'][key]).open('xb') as stream: np.save(stream,array,allow_pickle=False)
        del frame_train,frame_test
    elif binding['format']=='OmniAnomaly_machine_txt':
        train=np.loadtxt(inputs['train'],delimiter=','); test=np.loadtxt(inputs['test'],delimiter=',')
        labels=np.loadtxt(inputs['label'],delimiter=',')
        for key,array in [('train',train),('test',test),('label',labels)]:
            with (target/binding['output_filenames'][key]).open('xb') as stream: np.save(stream,array,allow_pickle=False)
    else: raise ValueError('Only explicitly registered full native data transformations supported')
    if (train.ndim!=2 or test.ndim!=2 or train.shape[1]!=binding['features'] or test.shape[1]!=binding['features']
            or labels.reshape(-1).shape[0]!=test.shape[0] or not np.isin(labels,[0,1]).all()
            or not np.isfinite(np.nan_to_num(train)).all() or not np.isfinite(np.nan_to_num(test)).all()):
        raise ValueError('Full numeric matrices, feature counts, binary point labels and exact row binding required')
    if binding.get('expected_shapes') and any(list(array.shape)!=binding['expected_shapes'][key]
            for key,array in [('train',train),('test',test)]):
        raise ValueError('All registered original rows and channels required without trimming')
    if binding['format']!='PSM_timestamp_csv':
        for key,expected in [('train',train),('test',test),('label',labels)]:
            if not np.array_equal(np.load(target/binding['output_filenames'][key],allow_pickle=False),expected,equal_nan=True):
                raise ValueError('Full saved native input array differs from original payload')
    return target,train,test,labels


def audit_loaders(config, model_config, binding, target, raw_train, raw_test, labels):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    source=native.verify(model_config)/'data_factory/data_loader.py'
    spec=importlib.util.spec_from_file_location('_iia_shcl_native_data_loader',str(source))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    constructor=getattr(module,binding['native_class'])
    clean_train=np.nan_to_num(raw_train) if binding['format']=='PSM_timestamp_csv' else raw_train
    clean_test=np.nan_to_num(raw_test) if binding['format']=='PSM_timestamp_csv' else raw_test
    scaler=StandardScaler().fit(clean_train)
    scaled_train=scaler.transform(clean_train); scaled_test=scaler.transform(clean_test)
    split=int(len(raw_train)*.8); rows=[]
    data_path=str(target/binding['path_prefix']) if binding.get('path_prefix') else str(target)
    for profile in config['profiles']:
        if profile['dataset']!=binding['dataset']: continue
        win=profile['parameters']['win_size']
        for mode in ('train','val','test','thre'):
            loader=constructor(data_path,win,1,mode)
            for field,expected in [('train',scaled_train[:split]),('val',scaled_train[split:]),('test',scaled_test),('test_labels',labels)]:
                if not np.array_equal(getattr(loader,field),expected,equal_nan=True):
                    raise ValueError('Full original native loader field differs: '+field)
            if not np.array_equal(loader.scaler.mean_,scaler.mean_) or not np.array_equal(loader.scaler.scale_,scaler.scale_):
                raise ValueError('Full original train-fit scaler state differs')
            values=scaled_train[:split] if mode=='train' else scaled_train[split:] if mode in ('val','thre') else scaled_test
            step=1 if mode in ('train','val') else win
            expected_windows=(len(values)-win)//step+1
            if len(loader)!=expected_windows or expected_windows<=0:
                raise ValueError('Every source sliding/nonoverlap window required')
            for index in (0,expected_windows-1):
                data,label=loader[index]; start=index*step
                expected_label=labels[start:start+win] if mode=='test' else labels[:win]
                if (not np.array_equal(data,values[start:start+win].astype(np.float32),equal_nan=True)
                        or not np.array_equal(label,expected_label.astype(np.float32))):
                    raise ValueError('Original boundary window or source label slicing differs')
            covered=(expected_windows-1)*step+win
            rows.append(dict(window=win,mode=mode,full_native_windows=expected_windows,full_input_rows=len(values),
                native_covered_prefix_rows=covered,source_dropped_tail_rows=len(values)-covered,
                complete_native_arrays_and_scaler_bitwise_verified=True,boundary_windows_and_labels_bitwise_verified=True))
            del loader; gc.collect()
    return dict(dataset=binding['dataset'],full_raw_train_shape=list(raw_train.shape),full_raw_test_shape=list(raw_test.shape),
        original_label_shape=list(labels.shape),source_train_validation_split_row=split,
        scaler_fit_includes_original_validation_rows=True,test_rows_used_for_scaler_fit=False,
        train_and_validation_return_source_dummy_test_label_slice=True,
        native_loaders=rows,full_native_loaders_audited=len(rows),
        data_path=data_path, output_files=[dict(path=str(target/name),sha256=native.sha(target/name))
            for name in binding['output_filenames'].values()],
        raw_and_feature_row_order_unchanged=True,
        author_data_version_equivalence=binding['author_version_boundary'], strict_grouped_TAB_protocol=False)


def worker(config, config_path):
    model_config=verify(config)
    if Path(config['view_root']).exists(): raise FileExistsError('Preserve every data preparation attempt')
    records=[]
    for binding in config['datasets']:
        target,train,test,labels=prepare_view(config,binding)
        row=audit_loaders(config,model_config,binding,target,train,test,labels)
        records.append(row)
        exclusive(Path(config['view_root'])/(binding['dataset']+'_data_audit.json'),row)
        print(json.dumps(dict(dataset=binding['dataset'],full_native_loaders=row['full_native_loaders_audited'],
            train_shape=list(train.shape),test_shape=list(test.shape))),flush=True)
        del train,test,labels; gc.collect()
    if sum(row['full_native_loaders_audited'] for row in records)!=52:
        raise ValueError('All13 profiles times4 original native modes required')
    verify(config)
    exclusive(Path(config['view_root'])/'full_data_preparation_audit.json',dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(),configuration=config_path,
        configuration_sha256=native.sha(native.ROOT/config_path),environment=data_environment(),
        full_datasets=5,full_native_loader_profiles=13,full_native_loader_modes_audited=52,records=records,
        preserved_raw_files_unchanged=True,original_author_loader_source_changed=False,
        preprocessing='Independent preserved inputs: PSM timestamp removal, SWaT CSV numeric-array conversion, '
            'SMD machine-3-11 TXT-array conversion, byte copies for MSL/SMAP. No row removal or extra normalization.',
        original_training_executed=False,paper_performance_reproduced=False,
        all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); config=native.read(native.ROOT/args.config)
    if not args.worker: raise ValueError('Guarded independent data worker required')
    worker(config,args.config)


if __name__=='__main__':main()
