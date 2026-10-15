"""Independent all1949 original anomaly cache identity and full-coordinate audit."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as producer
from scripts.flow_matching import verify_moment_icm_PILE_classification_v2 as arrays
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=producer.ROOT,producer.read,producer.sha,producer.write
MODULE='scripts.flow_matching.verify_moment_icm_PILE_anomaly_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_PILE_all1949_anomaly_closure.2026-10-15_v1.json'
REPORT='full1949_anomaly_cache_closure.json'


def raw_records(record):
    """Separate parser/count/record-identity reconstruction, without source data class."""
    import numpy as np
    path=ROOT/record['path']; before=path.stat()
    if sha(path)!=record['sha256'] or before.st_size!=record['size_bytes']:raise ValueError('Original raw CSV changed')
    ids=[];first=None
    with path.open('r',encoding='utf-8',newline='') as stream:
        for index,row in enumerate(csv.reader(stream)):
            if not row:continue
            if len(row)!=2:raise ValueError('Exactly2source CSV columns required')
            if first is None:first=row
            else:ids.append(index)
    after=path.stat()
    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):raise ValueError('Raw source changed during audit')
    try:numeric=all(np.isfinite(float(x)) for x in first)
    except (TypeError,ValueError):numeric=False
    return np.asarray(ids,dtype=np.int64),dict(first_record=first,first_record_numeric_finite=numeric,
        parsed_source_rows=len(ids),raw_nonempty_records=len(ids)+1,sha256=record['sha256'])


def expected_splits(job,rows,train_rows=None):
    family=job['family']; n=len(rows)
    if family in ['NASA-MSL','NASA-SMAP']:
        if train_rows is None:raise ValueError('Full paired NASA training raw identity required')
        fit=len(train_rows);border=int((.6/(.6+.1))*fit)
        return dict(train=train_rows[:border],val=train_rows[border:],test=rows),fit
    if family=='KDD21':
        official=int(Path(job['raw']['path']).name.split('.')[0].split('_')[4]); border=int((.6/(.6+.1))*official)
        return dict(train=rows[:border],val=rows[border:official],test=rows[official:]),border
    border=int(.6*n);test=n-int(.3*n)
    return dict(train=rows[:border],val=rows[border:test],test=rows[test:-1]),border


def close_split(directory,record,expected_rows,fit_rows,family,raw_path):
    import numpy as np
    directory=Path(directory);count=len(expected_rows)
    if (record['time_rows']!=count or record['source_dunder_len']!=count-255
            or record['source_length_gate_selected']!=(count>=256)
            or record['eligible_stride1_window_count']!=max(0,count-255)
            or record['source_scaler_fit_rows']!=fit_rows or record['source_CSV_path']!=raw_path
            or record['scaler_includes_NASA_validation_rows']!=(family in ['NASA-MSL','NASA-SMAP'])
            or record['source_general_test_omits_last_parsed_row']!=(family not in ['NASA-MSL','NASA-SMAP','KDD21'])
            or record['labels_saved_as_model_inputs'] or not record['all_source_signal_coordinates_byte_equal']
            or record['sequence_length']!=256 or record['stride']!=1 or record['channels']!=1):
        raise ValueError('Original split/scaler/header identity and complete source length gate required')
    names=dict(values='values_T.npy',rows='raw_CSV_record_indices.npy',mask='input_mask_window.npy')
    if {p.name for p in directory.glob('*.npy')}!=set(names.values()):raise ValueError('Exactly3label-free complete arrays required')
    scans={}
    for key,name in names.items():
        path=directory/name;before=path.stat()
        if sha(path)!=record['files'][key]['sha256']:raise ValueError('Complete source array hash changed')
        shape,dtype,offset=arrays.header(path)
        if list(shape)!=record['files'][key]['shape'] or str(dtype)!=record['files'][key]['dtype']:
            raise ValueError('Original array header changed')
        if key=='values':
            if shape!=(count,) or dtype!=np.dtype('float64'):raise ValueError('Full native float64 univariate vector required')
            if count:scans['signal']=arrays.scan(path,shape,dtype,offset,'signal',16*2**20)
        elif key=='rows':
            if shape!=(count,) or dtype!=np.dtype('int64'):raise ValueError('Full raw CSV record indices required')
            if not np.array_equal(np.load(path,allow_pickle=False),expected_rows):raise ValueError('Raw CSV record sequence changed')
        else:
            if shape!=(256,) or dtype!=np.dtype('float64') or not np.array_equal(np.load(path,allow_pickle=False),np.ones(256)):
                raise ValueError('Complete source shared observed mask required')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Cache array changed during closure')
    for key in ['source_scaler_mean','source_scaler_var','source_scaler_scale']:
        v=np.asarray(record[key])
        if v.shape!=(1,) or not np.isfinite(v).all():raise ValueError('Finite native source scaler required')
    return dict(time_rows=count,signal_coordinates=count,source_length_gate_selected=count>=256,
        eligible_windows=max(0,count-255),complete_arrays=3,raw_record_identity_equal=True,
        all_signal_coordinates_finite=True,constant_window_mask_verified=True,bounded_scan=scans,
        independent_normalized_signal_values_reconstructed=False)


def registration():
    previous=read(ROOT/producer.CONFIG)
    pins=[producer.CONFIG,'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_classification_v2.py',
        'tests/test_fm_moment_icm_PILE_anomaly_closure_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        producer_configuration=producer.CONFIG,sources=1949,families=18,splits=producer.SPLITS,
        required_cells=5847,required_arrays=17541,environment=previous['environment'],python=previous['python'],
        settings=previous['settings'],resource_lock=previous['resource_lock'],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_PILE_all1949_anomaly_closure_v1',
        state_root='experiments/runs/fm_moment_icm_PILE_all1949_anomaly_closure_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        boundary=('All1949source references/18guardedfamilyworkers/all5847originalsplits/17541arrays. '
            'Independent originalCSVrecord/header/count/pair/split/scalerfitextent/eligibility audit, '
            'fullraw/cache hashes,nativefloat64headers,allfinitecoordinates,allrawrecordindices '
            'andsharedmask. Original numericfirstrow loss/NASAvalidationfit/generaltestlastrow '
            'drop retained as source controls. No independent complete normalization/interpolation '
            'rewrite,neuralfit/inference/SVM,TAB/historicalcertification. Same16/21/16/3GiBdata '
            'resourcepolicy,allfailurespreservedwithoutretry,rawdatauntouched.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen anomaly closure input changed: '+pin['path'])
    previous=read(ROOT/config['producer_configuration']);jobs=producer.verify(previous)
    if (config['sources']!=1949 or config['families']!=18 or config['required_cells']!=5847
            or config['required_arrays']!=17541 or config['splits']!=producer.SPLITS
            or config['settings']!=previous['settings'] or config['environment']!=previous['environment']):
        raise ValueError('Full anomaly sources/splits/arrays and original data policy required')
    return previous,jobs


def worker(config):
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact source data environment required')
    previous,jobs=verify(config);state=read(ROOT/previous['state_root']/'status.json')
    families=sorted({j['family'] for j in jobs})
    if sorted(state['completed_families'])!=families or state['failed_preserved']:raise ValueError('All18successful complete original families required')
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    publish=status_writer(output/'progress.json',write);results=[];parents={}
    for family in families:
        path=Path(previous['artifact_root'])/family/'family_preparation.json';proof=read(path)
        guard_path=ROOT/previous['state_root']/'attempts'/family/'guard_receipt.json';guard=read(guard_path)
        subset=[j for j in jobs if j['family']==family]
        expected=[previous['python'],'-B','-X','utf8','-u','-m',producer.MODULE,'--family',family]
        if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
                or guard['proof_sha256']!=sha(path) or guard['exact_child_handle']['command']!=expected
                or not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/producer.CONFIG)
                or proof['failed_preserved'] or proof['environment']!=actual
                or proof['required_sources']!=len(subset)
                or [r['id'] for r in proof['records']]!=[j['id'] for j in subset]):
            raise ValueError('Complete actual family source/guard/scope chain required')
        parents[family]=dict(proof_sha256=sha(path),guard_sha256=sha(guard_path),records={r['id']:r for r in proof['records']})
    for job in jobs:
        directory=Path(previous['artifact_root'])/job['family']/job['id']
        proof_path=directory/'pretraining_preparation.json';proof=read(proof_path)
        if (not proof['passed'] or proof['job']!=job or proof['configuration_sha256']!=sha(ROOT/producer.CONFIG)
                or sha(proof_path)!=parents[job['family']]['records'][job['id']]['sha256']
                or not proof['original_complete_class_AST_reused'] or not proof['original_header_split_scaler_unmodified']
                or proof['neural_fit_inference_SVM']!=0 or proof['historical_ICM_pipeline_certified']):
            raise ValueError('Exact original source preparation identity required')
        rows,raw=raw_records(job['raw']);train_rows=train_raw=None
        if job['train_raw']:train_rows,train_raw=raw_records(job['train_raw'])
        for actual_raw,original_audit in [(raw,proof['raw_CSV_audit']),(train_raw,proof['paired_NASA_train_audit'])]:
            if actual_raw is None:
                if original_audit is not None:raise ValueError('Unexpected nonNASA paired raw')
            elif (actual_raw['first_record']!=original_audit['first_CSV_record']
                    or actual_raw['first_record_numeric_finite']!=original_audit['first_record_numeric_finite']
                    or actual_raw['parsed_source_rows']!=original_audit['parsed_rows_with_original_header_infer']
                    or actual_raw['sha256']!=original_audit['sha256']):raise ValueError('Independent raw header/count audit differs')
        split_rows,fit_rows=expected_splits(job,rows,train_rows);splits={}
        for split in producer.SPLITS:
            raw_path=job['train_raw']['path'] if job['train_raw'] and split!='test' else job['raw']['path']
            splits[split]=close_split(directory/split,proof['splits'][split],split_rows[split],fit_rows,job['family'],raw_path)
        results.append(dict(id=job['id'],family=job['family'],raw=raw,NASA_paired_train_raw=train_raw,
            cache_directory=str(directory),source_proof_sha256=sha(proof_path),splits=splits))
        publish(output/'progress.json',dict(completed_sources=len(results),required_sources=1949))
    write(output/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),sources=1949,families=18,cells=5847,arrays=17541,
        full_signal_coordinates=sum(c['signal_coordinates'] for r in results for c in r['splits'].values()),
        source_training_gate_selected=sum(r['splits']['train']['source_length_gate_selected'] for r in results),
        source_training_windows=sum(r['splits']['train']['eligible_windows'] for r in results),
        numeric_first_record_omitted_sources=sum(r['raw']['first_record_numeric_finite'] for r in results),
        NASA_pair_references=sum(r['NASA_paired_train_raw'] is not None for r in results),records=results,
        family_proof_chain=parents,environment=actual,independent_normalization_reconstructed=False,
        strict_temporal_independence_certified=False,historical_ICM_pipeline_certified=False,
        neural_fit_inference_SVM=0,all_original_experiments_complete=False))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing anomaly closure owner preserved')
    publish=status_writer(base/'status.json',write);state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(value=None,**extra):state.update(value or {});state.update(extra);publish(base/'status.json',state)
    try:
        if (base/'worker_launch.json').exists():raise ValueError('Existing closure attempt preserved;no retry')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],lambda v:update(v,state='waiting_original_data_memory_policy'),poll_seconds=10):
            cmd=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=cmd)
                write(base/'worker_launch.json',handle);update(state='closing_all1949_anomaly_sources',active_pid=p.pid)
                code,usage=capped_wait(p,config,api,lambda v:update(v,state='closing_all1949_anomaly_sources'))
            proof=Path(config['artifact_root'])/REPORT
            passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
            update(state='all1949_anomaly_cache_closure_passed' if passed else 'failed_preserved_without_retry',active_pid=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen anomaly closure registration')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,sources=1949,cells=5847,arrays=17541,torch_imported='torch' in sys.modules)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
