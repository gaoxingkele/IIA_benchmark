"""Prepare all1949 source anomaly references, preserving the original parser.

No header/split/scaler correction is applied to the source reproduction track.
Compact vectors retain raw CSV record identities; labels are not model inputs.
"""
from __future__ import annotations
import argparse
import ast
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types
from typing import Optional
import warnings

from scripts.flow_matching import prepare_moment_icm_PILE_long_forecast_v1 as storage
from scripts.flow_matching import prepare_moment_icm_classification_v1 as components
from scripts.flow_matching.moment_icm_pretraining_inputs_v1 import ROOT, read, sha, write
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

MODULE = 'scripts.flow_matching.prepare_moment_icm_PILE_anomaly_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_PILE_all1949_anomaly_pretraining.2026-10-15_v1.json'
SPLITS = ['train', 'val', 'test']


def source_anomaly_class(directory):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    path = Path(directory) / 'moment/data/anomaly_detection_datasets.py'
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(n, ast.ClassDef) and n.name == 'AnomalyDetectionDataset']
    if len(nodes) != 1: raise ValueError('One complete original anomaly source class required')
    namespace = dict(np=np, pd=pd, os=os, StandardScaler=StandardScaler, Optional=Optional,
        warnings=warnings, TaskDataset=object, DataSplits=types.SimpleNamespace, TimeseriesData=types.SimpleNamespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['AnomalyDetectionDataset']


def source_dataset(cls, path, split):
    if split not in SPLITS: raise ValueError('Explicit original split required')
    return cls(seq_len=256, full_file_path_and_name=Path(path).as_posix(), data_split=split,
        scale=True, data_stride_len=1, task_name='pre-training', train_ratio=.6, val_ratio=.1,
        test_ratio=.3, output_type='univariate', random_seed=13,
        downsampling_factor=None, min_length=None)


def csv_identity(record):
    """Audit physical CSV records independently of pandas' inferred header."""
    import numpy as np
    path = ROOT / record['path']
    if path.stat().st_size != record['size_bytes'] or sha(path) != record['sha256']:
        raise ValueError('Complete immutable original raw file required: ' + str(path))
    kept, first, columns = [], None, None
    with path.open('r', encoding='utf-8', newline='') as stream:
        for i, row in enumerate(csv.reader(stream)):
            if not row: continue  # pd.read_csv default skip_blank_lines
            if columns is None: columns = len(row)
            if len(row) != columns or columns != 2:
                raise ValueError('Unambiguous signal/label two-column CSV required')
            if first is None: first = row
            kept.append(i)
    if not first or len(kept) < 2: raise ValueError('Header-inferred source must retain data records')
    try:
        numeric_first = all(np.isfinite(float(x)) for x in first)
    except ValueError: numeric_first = False
    return dict(path=record['path'], sha256=record['sha256'], raw_nonempty_records=len(kept),
        parsed_rows_with_original_header_infer=len(kept)-1, first_CSV_record=first,
        first_record_numeric_finite=numeric_first, source_default_header_infer=True,
        source_discards_numeric_first_record=numeric_first), np.asarray(kept[1:], dtype=np.int64)


def original_split_rows(dataset, job, raw_rows, NASA_train_rows=None):
    family, split = job['family'], dataset.data_split
    if family in ['NASA-MSL','NASA-SMAP']:
        if NASA_train_rows is None: raise ValueError('Explicit paired NASA training-file row count required')
        if split == 'test': return raw_rows, NASA_train_rows
        # The class has overwritten length_timeseries with the selected split.
        total = len(raw_rows); end = int((.6/(.6+.1))*total)
        return raw_rows[:end] if split == 'train' else raw_rows[end:], total
    total = len(raw_rows)
    if family == 'KDD21':
        official = int(dataset.series.split('_')[4]); end = int((.6/(.6+.1))*official)
        bounds = dict(train=slice(0,end),val=slice(end,official),test=slice(official,None))
    else:
        end = int(.6*total); test_start = total-int(.3*total)
        bounds = dict(train=slice(0,end),val=slice(end,test_start),test=slice(test_start,-1))
    fit_rows = bounds['train'].stop
    return raw_rows[bounds[split]], fit_rows


def export_split(dataset, job, raw_rows, output, NASA_train_rows=None):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    signal = dataset.data
    if signal.ndim != 1 or dataset.n_channels != 1 or len(signal) != dataset.length_timeseries:
        raise ValueError('All original univariate source coordinates required')
    rows, fit_rows = original_split_rows(dataset, job, raw_rows, NASA_train_rows)
    if len(rows) != len(signal): raise ValueError('Every source signal coordinate must retain raw record identity')
    n = dataset.__len__()  # May be negative. Do not call len() on excluded short sources.
    eligible = len(signal) >= 256
    if n != len(signal)-255: raise ValueError('Original untrimmed stride1 window arithmetic required')
    files = {}
    for key, name, array in [('values','values_T.npy',signal),('rows','raw_CSV_record_indices.npy',rows),
                              ('mask','input_mask_window.npy',np.ones(256))]:
        if array.shape[0]: files[key] = storage.save_source_array(output/name,array)
        else:
            np.save(output/name,array,allow_pickle=False)
            files[key] = dict(sha256=sha(output/name),shape=list(array.shape),dtype=str(array.dtype),
                all_source_coordinates_byte_equal=True,empty_original_split=True)
    checked = sorted({0,n//2,n-1}) if eligible else []
    for index in checked:
        item = dataset[index]
        if (item.timeseries.shape != (1,256) or item.timeseries.tobytes() != signal[index:index+256].reshape(1,256).tobytes()
                or not np.array_equal(item.input_mask,np.ones(256)) or hasattr(item,'labels')):
            raise ValueError('Original pretraining getitem/axes/mask/label exclusion required')
    if not eligible and dataset[0] is not None: raise ValueError('Source short-series behavior changed')
    if int(dataset.scaler.n_samples_seen_) != fit_rows:
        raise ValueError('Exact original scaler fit extent required')
    return dict(time_rows=len(signal),source_dunder_len=n,source_length_gate_selected=eligible,
        eligible_stride1_window_count=max(0,n),sequence_length=256,stride=1,channels=1,
        source_CSV_path=job['train_raw']['path'] if job['family'] in ['NASA-MSL','NASA-SMAP'] and dataset.data_split!='test' else job['raw']['path'],
        source_scaler_fit_rows=fit_rows,source_scaler_mean=dataset.scaler.mean_.tolist(),
        source_scaler_var=dataset.scaler.var_.tolist(),source_scaler_scale=dataset.scaler.scale_.tolist(),
        scaler_includes_NASA_validation_rows=job['family'] in ['NASA-MSL','NASA-SMAP'],
        source_full_CSV_cubic_interpolation_precedes_time_split=True,
        source_general_test_omits_last_parsed_row=job['family'] not in ['NASA-MSL','NASA-SMAP','KDD21'],
        all_source_signal_coordinates_byte_equal=True,window_endpoint_indices_checked=checked,
        all_source_windows_independently_replayed=False,labels_saved_as_model_inputs=False,
        files=files,strict_temporal_independence_certified=False)


def registration():
    scope = read(ROOT/storage.SCOPE)
    previous_path = 'configs/reproducibility/fm_moment_ICM_PILE_all9_long_forecast_pretraining.2026-10-15_v1.json'
    previous = read(ROOT/previous_path)
    registry = {r['path']:r for r in read(ROOT/scope['integrity'])['records']}
    jobs = []
    for path in scope['all_source_getter_paths']['anomaly']:
        p=Path(path);family=p.parent.name
        raw=registry[path]; train_raw=None
        if family in ['NASA-MSL','NASA-SMAP']:
            pair=(p.parent/(p.name.split('.')[0]+'.train.out')).as_posix()
            train_raw=registry[pair]
        jobs.append(dict(id=raw['id'],family=family,raw=raw,train_raw=train_raw))
    pins=[storage.SCOPE,scope['integrity'],previous_path,
        'scripts/flow_matching/prepare_moment_icm_PILE_anomaly_v1.py',
        'tests/test_fm_moment_icm_PILE_anomaly_v1.py',
        scope['source_root']+'/moment/data/anomaly_detection_datasets.py']
    pins += [p['path'] for p in previous['source_receipts']]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=scope['paper_citation'],
        source_root=scope['source_root'],complete_input_scope=storage.SCOPE,
        required_sources=1949,required_families=18,jobs=jobs,splits=SPLITS,
        track='unaltered_related_source_header_infer_and_time_splits',sequence_length=256,
        environment=previous['environment'],python=previous['python'],settings=previous['settings'],
        resource_lock=previous['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all1949_anomaly_pretraining_v1',
        state_root='experiments/runs/fm_moment_icm_PILE_all1949_anomaly_pretraining_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(pins))],
        boundary=('All1949getterreferences/all18families/all3splits,includingexcludedshortsources. '
            'Pinned complete source class AST reused;no header/slice/scaler correction,downsampling, '
            'precision/window/budget reduction. Audit numericfirstrecord removal per actual file, '
            'NASA paired train/testrawidentities and fulltrainfile scalerfit(includingvalidation), '
            'KDDofficialboundary and general-lastrowdrop. Compact complete1Dsignals/rawCSVrecordIDs '
            '/shared256all1mask;no pretraining labels. Shortsources keep original negative len arithmetic '
            'and are ineligible,not padded into training. No historicalpaperpipeline/TAB/leakage '
            'certification from file names. Nonneural16physical/21commit/16emergency/3GiBtree policy '
            'unchanged. No raw/oldcache overwrite,failedattempt retry,NNpretraining/inference/SVM '
            'or performance claim;strictheader/restoredsplit controls remain separate future tasks.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']: raise ValueError('Frozen source changed: '+pin['path'])
    scope=read(ROOT/config['complete_input_scope']); jobs=config['jobs']
    if (len(jobs)!=config['required_sources'] or config['required_sources']!=1949
            or len({j['family'] for j in jobs})!=18 or config['required_families']!=18
            or len({j['id'] for j in jobs})!=1949
            or [j['raw']['path'] for j in jobs]!=scope['all_source_getter_paths']['anomaly']
            or config['splits']!=SPLITS or config['sequence_length']!=256
            or config['environment']!=components.ENVIRONMENT
            or config['track']!='unaltered_related_source_header_infer_and_time_splits'
            or config['settings']!=dict(minimum_available_memory_bytes=16*2**30,
                minimum_commit_headroom_bytes=21*2**30,emergency_commit_headroom_bytes=16*2**30,
                maximum_worker_rss_bytes=3*2**30,maximum_worker_private_bytes=3*2**30)):
        raise ValueError('Complete original1949sources,18families and frozen data policy required')
    for job in jobs:
        if (job['family'] in ['NASA-MSL','NASA-SMAP']) != (job['train_raw'] is not None):
            raise ValueError('All and only NASA references must bind paired train originals')
    return jobs


def worker(config, family, jobs):
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']: raise ValueError('Exact source parser environment required')
    cls=source_anomaly_class(ROOT/config['source_root'])
    output=Path(config['artifact_root'])/family;output.mkdir(parents=True,exist_ok=False)
    progress=status_writer(output/'progress.json',write); records=[];failed={}
    for job in jobs:
        folder=output/job['id']
        try:
            raw,rows=csv_identity(job['raw']);train_raw,train_rows=(csv_identity(job['train_raw']) if job['train_raw'] else (None,None))
            folder.mkdir(parents=True,exist_ok=False);splits={}
            for split in SPLITS:
                data=source_dataset(cls,ROOT/job['raw']['path'],split)
                identities=train_rows if job['train_raw'] and split!='test' else rows
                splits[split]=export_split(data,job,identities,folder/split,
                    NASA_train_rows=len(train_rows) if train_rows is not None else None)
                del data;gc.collect()
            proof=folder/'pretraining_preparation.json'
            write(proof,dict(passed=True,job=job,configuration_sha256=sha(ROOT/CONFIG),
                raw_CSV_audit=raw,paired_NASA_train_audit=train_raw,splits=splits,
                original_complete_class_AST_reused=True,original_header_split_scaler_unmodified=True,
                historical_ICM_pipeline_certified=False,neural_fit_inference_SVM=0))
            records.append(dict(id=job['id'],proof=str(proof),sha256=sha(proof)))
        except Exception as error:
            failed[job['id']]=dict(type=type(error).__name__,message=str(error),raw_path=job['raw']['path'])
            folder.mkdir(parents=True,exist_ok=True);write(folder/'failed_preserved.json',failed[job['id']])
        progress(output/'progress.json',dict(family=family,completed_sources=len(records),required=len(jobs),failed_preserved=failed))
    write(output/'family_preparation.json',dict(passed=not failed and len(records)==len(jobs),family=family,
        configuration_sha256=sha(ROOT/CONFIG),records=records,failed_preserved=failed,environment=actual,
        required_sources=len(jobs),neural_fit_inference_SVM=0,all_original_experiments_complete=False))


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Existing anomaly data owner preserved')
    publish=status_writer(base/'status.json',write);completed=[];failed={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,required_sources=1949,required_families=18)
    def update(value=None,**extra):
        state.update(value or {});state.update(extra)
        state.update(completed_families=list(completed),failed_preserved=dict(failed))
        publish(base/'status.json',state)
    try:
        for family in sorted({j['family'] for j in jobs}):
            attempt=base/'attempts'/family;proof=Path(config['artifact_root'])/family/'family_preparation.json'
            try:
                if attempt.exists() or proof.parent.exists(): raise ValueError('Existing family attempt preserved;no automatic retry')
                with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],
                    lambda value:update(value,state='waiting_original_data_memory_policy',active_pid=None,family=family),poll_seconds=10):
                    attempt.mkdir(parents=True,exist_ok=False)
                    cmd=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--family',family]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=cmd)
                        write(attempt/'launch.json',handle);update(state='preparing_original_anomaly_sources',active_pid=p.pid,family=family)
                        code,usage=capped_wait(p,config,api,lambda value:update(value,state='preparing_original_anomaly_sources'))
                    passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                        proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed: raise ValueError('Original family preparation failed;preserved without retry')
                completed.append(family)
            except Exception as error: failed[family]=dict(type=type(error).__name__,message=str(error))
            update(state='scanning_next_anomaly_family',active_pid=None,family=None)
        update(state='all18_anomaly_family_attempts_terminal',active_pid=None,family=None)
    finally: owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--family')
    args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists(): raise FileExistsError('Preserve frozen anomaly registration')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream: json.dump(config,stream,indent=2);stream.write('\n')
    else: config=read(ROOT/CONFIG);verify(config)
    if args.probe_only: print(json.dumps(dict(passed=True,sources=1949,families=18,splits=3,torch_imported='torch' in sys.modules)))
    elif args.family:
        jobs=[j for j in config['jobs'] if j['family']==args.family]
        if not jobs: raise ValueError('Exact registered family required')
        worker(config,args.family,jobs)
    else: controller(config,config['jobs'])


if __name__=='__main__': main()
