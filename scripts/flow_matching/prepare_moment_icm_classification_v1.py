"""Full117dataset source-derived preprocessing for future Tiny/ICM classification.

Only private AST copies of pinned MOMENT components execute. Keep every official
TRAIN/TEST instance and channel, including split-specific source standardizers.
This is a transductive source control, not a strict train-only preprocessing track.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import ROOT,read,sha,write
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

MODULE='scripts.flow_matching.prepare_moment_icm_classification_v1'
PAPER_SCOPE='configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json'
ENVIRONMENT=dict(python='3.10.21',numpy='1.25.2',pandas='2.2.2',scipy='1.9.3',**{'scikit-learn':'1.5.2'})


def source_components(directory):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    directory=Path(directory)
    def module(relative):
        path=directory/relative
        namespace=dict(__name__='_isolated_moment_preprocessing',__file__=str(path))
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
        return namespace
    loader=module('moment/data/load_data.py')
    utility=module('moment/utils/data.py')
    path=directory/'moment/data/classification_datasets.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='ClassificationDataset')
        or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATASETS_WITHOUT_NORMALIZATION' for t in n.targets))]
    if len(nodes)!=2:raise ValueError('Exact pinned classification class and normalization exclusions required')
    namespace=dict(np=np,pd=pd,os=os,StandardScaler=StandardScaler,logging=logging,
        TaskDataset=object,DataSplits=types.SimpleNamespace,TimeseriesData=types.SimpleNamespace,
        load_from_tsfile=loader['load_from_tsfile'],**{k:utility[k] for k in ['interpolate_timeseries','upsample_timeseries','downsample_timeseries']})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    script=directory/'scripts/zero_shot/unsupervised_representation_learning_multivariate.py'
    tree=ast.parse(script.read_text(encoding='utf-8'));small=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['SMALL_IMAGE_DATASETS','SMALL_SPECTRO_DATASETS'] for t in node.targets):
            small+=ast.literal_eval(node.value)
    if len(small)!=13:raise ValueError('All13 original interpolation exceptions required')
    return namespace['ClassificationDataset'],small,namespace['DATASETS_WITHOUT_NORMALIZATION']


def source_dataset(cls,path,dataset,small,split,train_ratio=1.,val_ratio=0.):
    return cls(seq_len=256,full_file_path_and_name=Path(path).as_posix(),data_split=split,
        scale=True,task_name='classification',train_ratio=train_ratio,val_ratio=val_ratio,
        test_ratio=0.,output_type='univariate',upsampling_pad_direction='backward',
        upsampling_type='interpolate' if dataset in small else 'pad',
        downsampling_type='interpolate',pad_mode='constant',pad_constant_values=0)


def export_split(dataset,expected_instances,output):
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    channels=dataset.n_channels
    if len(dataset)!=expected_instances*channels:raise ValueError('Every original instance and channel required')
    first=dataset[0]
    values=np.lib.format.open_memmap(output/'values.npy',mode='w+',dtype=first.timeseries.dtype,shape=(expected_instances,channels,256))
    masks=np.lib.format.open_memmap(output/'input_mask.npy',mode='w+',dtype=first.input_mask.dtype,shape=(expected_instances,256))
    labels=np.empty(expected_instances,dtype=np.int64)
    checked_coordinates=0
    for index in range(len(dataset)):
        item=first if index==0 else dataset[index];case,channel=divmod(index,channels)
        signal=item.timeseries
        if signal.shape!=(1,256) or item.input_mask.shape!=(256,) or not np.isfinite(signal).all():
            raise ValueError('Full original256coordinate finite source transform required')
        if signal.dtype!=values.dtype or item.input_mask.dtype!=masks.dtype or not np.isin(item.input_mask,[0,1]).all():
            raise ValueError('Original source dtype and binary padding mask required')
        values[case,channel]=signal[0]
        label=int(np.asarray(item.labels).item())
        if channel==0:masks[case]=item.input_mask;labels[case]=label
        elif not np.array_equal(masks[case],item.input_mask) or labels[case]!=label:
            raise ValueError('All channels must share original instance mask and label')
        if not np.array_equal(values[case,channel],signal[0]):raise ValueError('Saved source coordinates differ')
        checked_coordinates+=256
    values.flush();masks.flush();shape=list(values.shape);dtype=str(values.dtype)
    del values,masks
    np.save(output/'labels.npy',labels,allow_pickle=False)
    np.save(output/'original_instance_rows.npy',np.arange(expected_instances,dtype=np.int64),allow_pickle=False)
    return dict(instances=expected_instances,channels=channels,sequence_length=256,values_shape=shape,
        values_dtype=dtype,source_coordinates_copied_and_checked=checked_coordinates,
        official_instance_order_and_all_channels_retained=True,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        source_scaler_fit_sample_shape=list(dataset.scaler.mean_.shape) if hasattr(dataset.scaler,'mean_') else None)


def verify(config):
    scope=read(ROOT/PAPER_SCOPE)
    if (config['paper_scope']!=PAPER_SCOPE or config['paper_scope_sha256']!=sha(ROOT/PAPER_SCOPE)
            or config['sequence_length']!=256 or config['environment']!=ENVIRONMENT
            or config['source_normalization']!='original_separate_full_TRAIN_and_TEST_fits'
            or config['train_val_merge']!='entire_official_TRAIN_after_source_train_val_concatenation'
            or config['settings']!=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=21*2**30,
                emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=3*2**30,maximum_worker_private_bytes=3*2**30)):
        raise ValueError('Full source-derived preprocessing and fixed independent data-worker policy required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Preprocessing input changed: '+pin['path'])
    names=list(dict.fromkeys(r['dataset'] for r in scope['all630_reported_targets'] if r['table'] in [6,7]))
    if [j['dataset'] for j in config['jobs']]!=names or len(names)!=117:raise ValueError('All26multivariate and91univariate datasets required')
    audit=read(ROOT/config['raw_audit'])
    records={(r['dataset'],r['split']):r for r in audit['files'] if 'table' in r}
    for job in config['jobs']:
        if set(job['splits'])!={'TRAIN','TEST'}:raise ValueError('Both full official splits required')
        for split,binding in job['splits'].items():
            raw=records[job['dataset'],split]
            if raw['status']!='verified' or binding!=dict(path=raw['logical_path'],sha256=raw['sha256'],instances=raw['observed_instances']):
                raise ValueError('Original complete raw split binding differs')
    return config['jobs']


def worker(config,job):
    import platform
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=ENVIRONMENT:raise ValueError('Exact registered preprocessing interpreter and packages required')
    cls,small,unscaled=source_components(ROOT/config['source_root'])
    output=Path(config['artifact_root'])/job['dataset'];output.mkdir(parents=True,exist_ok=False)
    records={}
    for name in ['TRAIN','TEST']:
        raw=job['splits'][name];path=ROOT/raw['path']
        if sha(path)!=raw['sha256']:raise ValueError('Original raw data changed')
        data=source_dataset(cls,path,job['dataset'],small,'train' if name=='TRAIN' else 'test')
        records[name]=export_split(data,raw['instances'],output/name)
        del data;gc.collect()
        if sha(path)!=raw['sha256']:raise ValueError('Raw data changed during complete transformation')
    write(output/'preprocessing.json',dict(passed=True,job=job,configuration_sha256=sha(ROOT/config['_config_path']),
        environment=actual,source_root=config['source_root'],records=records,
        original_data_standardizer_fitting_performed=job['dataset'] not in unscaled,
        source_TEST_standardizer_fit_preserved_for_diagnostic_control=True,
        strict_train_only_preprocessing=False,source_AST_reused=True,independent_preprocessing_value_reconstruction=False,
        neural_model_training_or_inference_or_SVM_performed=False,raw_data_overwritten=False,
        exact_historical_ICM_preprocessing_certified=False,all_original_experiments_complete=False))


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full classification preprocessing already has a live owner')
    publish=status_writer(base/'status.json',write)
    completed=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_datasets=117,
        required_official_splits=234,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw)
        state.update(completed_datasets=list(completed),completed_count=len(completed),failed_preserved=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat())
        publish(base/'status.json',state)
    try:
        for job in jobs:
            identity=job['dataset'];attempt=base/'attempts'/identity;proof=Path(config['artifact_root'])/identity/'preprocessing.json'
            try:
                if attempt.exists() or proof.parent.exists():raise ValueError('Prior preprocessing attempt preserved;no automatic retry')
                def waiting(value):update(value,state='waiting_data_preprocessing_memory',active_pid=None,dataset=identity)
                with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],waiting,poll_seconds=10):
                    attempt.mkdir(parents=True,exist_ok=False)
                    command=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--config',config['_config_path'],'--dataset',identity]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                        write(attempt/'launch.json',handle);update(state='running_full_original_data_preprocessing',active_pid=child.pid,dataset=identity)
                        def usage(value):update(dict(value,state='running_full_original_data_preprocessing'))
                        code,peaks=capped_wait(child,config,api,usage)
                    passed=code==0 and not peaks['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
                    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**peaks,
                        proof_sha256=sha(proof) if proof.exists() else None,exact_child_handle=handle))
                    if not passed:raise ValueError('Original full preprocessing failed;preserved without retry')
                completed.append(identity)
            except Exception as error:failed[identity]=str(error)
            update(state='scanning_next_original_classification_dataset',active_pid=None,dataset=None)
        update(state='all117_preprocessing_attempts_terminal_original_model_experiments_remain',active_pid=None,dataset=None)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--dataset');p.add_argument('--probe-only',action='store_true');args=p.parse_args()
    config=read(ROOT/args.config);jobs=verify(config);config['_config_path']=args.config
    if args.probe_only:print(json.dumps(dict(passed=True,datasets=len(jobs),splits=234,torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules)))
    elif args.dataset:worker(config,next(j for j in jobs if j['dataset']==args.dataset))
    else:controller(config,jobs)


if __name__=='__main__':main()
