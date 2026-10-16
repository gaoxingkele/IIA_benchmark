"""Read every registered industrial raw coordinate without invoking its preparer.

The CSV parser and NumPy primitives are shared dependencies; selection, labels,
group boundaries, scaler fitting and exact payload checks are independently
implemented here. No original file or prepared archive is rewritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.audit_industrial_full_raw_preprocessing_v1'
TEST='tests/test_fm_industrial_full_raw_preprocessing_v1.py'
CONFIG='configs/reproducibility/fm_industrial_full_raw_preprocessing.2026-10-16_v1.json'
SETTINGS='configs/experiments/fm_industrial_tsad_execution.v1.json'
MANIFEST='data/public_datasets/flow_matching/processed_industrial_tsad_v1/manifest.json'


def registration():
    paths=[MODULE.replace('.','/')+'.py',TEST,SETTINGS,MANIFEST]
    paths+=sorted({s['source_config'] for s in read(ROOT/SETTINGS)['datasets']})
    return dict(settings=SETTINGS,manifest=MANIFEST,required_datasets=['tep_classic','skab','pronto_day4','pronto_day2','pronto_day3'],
        report_path='F:/aicoding/IIA_Data/experiments/industrial_full_raw_preprocessing_v1/full5_reconstruction.json',
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in paths],
        citations=read(ROOT/SETTINGS)['citations'],no_original_or_prepared_data_writes=True,
        new_training_updates=0,new_neural_forwards=0,all_original_experiments_complete=False)


def normal_spans(labels):
    spans=[];start=None
    for i,label in enumerate(labels):
        if label=='Normal' and start is None:start=i
        if label!='Normal' and start is not None:spans.append((start,i));start=None
    if start is not None:spans.append((start,len(labels)))
    return spans


def disjoint(roles):
    if set(roles)!={'train','validation','test'} or any(not rows for rows in roles.values()):raise ValueError('Every full role must be nonempty')
    for role,rows in roles.items():
        seen={}
        for g in rows:
            a,b=g['source_interval']
            if a<0 or b<=a or b-a!=len(g['values']):raise ValueError('Full contiguous source interval required')
            if any(a<right and left<b for left,right in seen.get(g['source_path'],[])):raise ValueError('Repeated source coordinates')
            seen.setdefault(g['source_path'],[]).append((a,b))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if {g['source_path'] for g in roles[a]}&{g['source_path'] for g in roles[b]}:raise ValueError('A source file crosses roles')
        if {g['group_id'] for g in roles[a]}&{g['group_id'] for g in roles[b]}:raise ValueError('A day/run crosses roles')


def independently_standardize(roles):
    disjoint(roles)
    dimensions={g['values'].shape[1] for rows in roles.values() for g in rows if g['values'].ndim==2}
    if len(dimensions)!=1 or any(g['values'].ndim!=2 for rows in roles.values() for g in rows):raise ValueError('Common full feature matrix required')
    fit=np.concatenate([g['values'] for g in roles['train']]).astype(float)
    finite=np.where(np.isfinite(fit),fit,np.nan)
    medians=np.nanmedian(finite,axis=0)
    if not np.isfinite(medians).all():raise ValueError('A fit channel has no finite sample')
    cleaned={};repairs={}
    for role,groups in roles.items():
        cleaned[role]=[];repairs[role]=[]
        for group in groups:
            x=np.array(group['values'],dtype=float,copy=True,order='C');bad=~np.isfinite(x)
            repairs[role].append(int(bad.sum()))
            row,column=np.nonzero(bad);x[row,column]=medians[column]
            cleaned[role].append(x)
    fit=np.concatenate(cleaned['train'])
    mean=np.mean(fit,axis=0);std=np.std(fit,axis=0,ddof=0);scale=np.where(std>0,std,1.)
    normalized={role:[np.asarray((x-mean)/scale,dtype=np.float32) for x in groups] for role,groups in cleaned.items()}
    if not all(np.isfinite(x).all() for groups in normalized.values() for x in groups):raise ValueError('Every standardized value must be finite')
    return normalized,dict(median=medians.tolist(),mean=mean.tolist(),std=std.tolist(),constant_channels=np.flatnonzero(std==0).tolist(),
        repairs=repairs,fit_points=len(fit),fit_role='train_only_unique_source_intervals')


def exact_payload(name,computed,saved):
    computed,saved=np.asarray(computed),np.asarray(saved)
    if computed.dtype!=saved.dtype or computed.shape!=saved.shape or computed.tobytes(order='C')!=saved.tobytes(order='C'):
        raise ValueError('Full dtype/shape/coordinate bytes differ: '+name)
    return dict(name=name,dtype=str(computed.dtype),shape=list(computed.shape),elements=int(computed.size),
        sha256=hashlib.sha256(computed.tobytes(order='C')).hexdigest())


def raw_group(relative,run,values,labels,names,timestamps,start=0,stop=None,faults=None):
    stop=len(values) if stop is None else stop
    return dict(source_path=relative,group_id=run,entity_id=f'{run}[{start}:{stop}]',source_interval=[start,stop],
        values=np.asarray(values[start:stop]),labels=np.asarray(labels[start:stop],dtype=np.uint8),feature_names=list(names),
        timestamps=np.asarray(timestamps[start:stop],dtype=float),fault_labels=None if faults is None else np.asarray(faults[start:stop],dtype=str))


def reconstruct_groups(root,spec,c):
    roles=dict(train=[],validation=[],test=[])
    if spec['loader']=='tep_whole_runs':
        tests=sorted(p for p in root.glob(c['test_glob']) if p.stem not in c['exclude_test_stems'])
        if {p.stem for p in tests}!={f'd{i:02d}_te' for i in range(1,22)}:raise ValueError('All21 classic fault tests required')
        files=dict(train=[root/c['train_path']],validation=[root/spec['validation_path']],test=tests)
        names=[f'XMEAS_{i:02d}' for i in range(1,42)]+[f'XMV_{i:02d}' for i in range(1,12)]
        for role,paths in files.items():
            for p in paths:
                # Separate parser; Python float conversion reads the complete text.
                x=np.asarray([[float(v) for v in line.split()] for line in p.read_text().splitlines() if line.strip()],dtype=float)
                if x.ndim!=2:raise ValueError('TEP must be a complete matrix')
                if x.shape[1]!=52:
                    if x.shape[0]!=52:raise ValueError('A TEP axis must contain52 variables')
                    x=x.T
                y=np.asarray([role=='test' and i>=c['fault_start'] for i in range(len(x))],dtype=np.uint8)
                relative=p.relative_to(root).as_posix();roles[role].append(raw_group(relative,relative,x,y,names,np.arange(len(x),dtype=float)*c['sample_period']))
    elif spec['loader']=='skab_whole_experiments':
        tests=sorted({p for pattern in c['test_globs'] for p in root.glob(pattern)})
        val=root/spec['validation_path']
        if len(tests)!=spec['expected_test_entities']+1 or val not in tests:raise ValueError('All SKAB experiments and fixed validation required')
        for role,paths in dict(train=[root/c['train_path']],validation=[val],test=[p for p in tests if p!=val]).items():
            for p in paths:
                frame=pd.read_csv(p,sep=';');names=[x for x in frame.columns if x not in {'datetime','anomaly','changepoint'}]
                x=frame[names].to_numpy(dtype=float)
                if not np.isfinite(x).all():raise ValueError('SKAB full finite sensors required')
                y=frame['anomaly'].to_numpy(dtype=float) if 'anomaly' in frame else np.zeros(len(frame))
                if not np.isin(y,[0,1]).all():raise ValueError('SKAB labels must remain native binary points')
                times=pd.to_datetime(frame['datetime'],utc=True,errors='raise').astype('int64').to_numpy(dtype=float)/1e9
                relative=p.relative_to(root).as_posix();roles[role].append(raw_group(relative,relative,x,y,names,times))
    elif spec['loader']=='pronto_whole_day':
        paths={Path(p).stem:p for p in c['paths']}
        for role,selection in [('train','training_day'),('validation','validation_day'),('test','test_day')]:
            relative=paths[spec[selection]];frame=pd.read_csv(root/relative)
            names=[x for x in frame.columns[c['alarm_column_count']:] if x!=c['label_column']]
            if frame.columns.duplicated().any() or len(names)!=17:raise ValueError('All17 identically ordered process channels required')
            alarms=frame.iloc[:,:c['alarm_column_count']].to_numpy()
            if not np.isin(alarms,[0,1]).all():raise ValueError('Native binary alarm fields must be preserved and excluded from process values')
            x=frame[names].to_numpy();faults=np.asarray(frame[c['label_column']].astype(str).to_numpy(),dtype=str)
            if not np.isfinite(x).all() or np.any(np.char.str_len(faults)==0):raise ValueError('Complete finite process and native fault labels required')
            spans=[(0,len(frame))] if role=='test' else normal_spans(faults)
            for a,b in spans:roles[role].append(raw_group(relative,Path(relative).stem,x,faults!='Normal',names,
                np.arange(len(frame))*c['sample_period_seconds'],a,b,faults))
    else:raise ValueError('Unregistered industrial schema')
    if len(roles['test'])!=spec['expected_test_entities']:raise ValueError('Full held-out entity roster required')
    if len({tuple(g['feature_names']) for rows in roles.values() for g in rows})!=1:raise ValueError('Feature order changed between roles')
    if any(g['values'].shape[1]!=c.get('features',17) for rows in roles.values() for g in rows):raise ValueError('Complete feature count required')
    return roles


def audit_dataset(root,spec,dataset):
    c=read(root/spec['source_config'])
    for p in dataset['source_files']:
        if sha(root/p['path'])!=p['sha256']:raise ValueError('Registered raw source bytes changed')
    if sha(root/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Registered prepared archive changed')
    roles=reconstruct_groups(root,spec,c);normalized,scaler=independently_standardize(roles)
    if scaler!=dataset['scaler']:raise ValueError('Raw TRAIN-only scaler differs: '+spec['id'])
    raw_set={g['source_path'] for groups in roles.values() for g in groups}
    if raw_set!={p['path'] for p in dataset['source_files']}:raise ValueError('Actual full raw file roster differs')
    arrays=[];expected=set();counts={};label_counts={};validation_labels=[]
    with np.load(root/dataset['input_npz'],allow_pickle=False) as saved:
        for role,groups in roles.items():
            meta=dataset['training_groups' if role=='train' else 'validation_groups' if role=='validation' else 'entities']
            if len(meta)!=len(groups):raise ValueError('Complete role/group roster differs')
            counts[role]=sum(len(g['values']) for g in groups);label_counts[role]=sum(int(g['labels'].sum()) for g in groups)
            for i,(g,x,m) in enumerate(zip(groups,normalized[role],meta)):
                prefix={'train':'t','validation':'v','test':'e'}[role]+f'{i:03d}'
                fields={k:g[k] for k in ['group_id','entity_id','source_path','source_interval','feature_names']}
                fields.update(array_prefix=prefix,points=len(x),features=x.shape[1],anomaly_points=int(g['labels'].sum()))
                if role=='test':fields['test_points']=len(x)
                if fields!=m:raise ValueError('Source group identity or coordinates differ')
                payload={prefix+('_test' if role=='test' else '_values'):x,prefix+'_labels':g['labels'],prefix+'_timestamps':g['timestamps'],
                    prefix+'_source_rows':np.arange(*g['source_interval'])}
                if g['fault_labels'] is not None:payload[prefix+'_fault_labels']=g['fault_labels']
                for key,value in payload.items():expected.add(key);arrays.append(exact_payload(key,value,saved[key]))
                if role=='validation':validation_labels.append(g['labels'])
        if set(saved.files)!=expected:raise ValueError('An original archive member is omitted or fabricated')
    for p in dataset['source_files']:
        if sha(root/p['path'])!=p['sha256']:raise ValueError('Raw data changed during read-only audit')
    if sha(root/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Prepared archive changed during read-only audit')
    return dict(dataset=spec['id'],passed=True,raw_source_files=dataset['source_files'],prepared_archive_sha256=dataset['input_sha256'],
        role_points=counts,role_positive_points=label_counts,role_group_counts={role:len(groups) for role,groups in roles.items()},
        feature_count=len(roles['train'][0]['feature_names']),array_count=len(arrays),all_array_payloads=arrays,
        independently_recomputed_scaler=scaler,raw_files_disjoint_across_roles=True,all_registered_arrays_bitwise_identical=True,
        validation_contains_native_anomalies=bool(label_counts['validation']),normal_selection_is_label_assisted=spec['loader']=='pronto_whole_day',
        independent_software_libraries_claim=False,shared_dependencies=['NumPy arithmetic','pandas CSV parsing for SKAB and PRONTO'],
        original_or_prepared_data_rewritten=False,new_neural_forwards=0,new_training_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG);args=parser.parse_args()
    c=read(ROOT/args.config)
    if c!=registration():raise ValueError('Frozen complete industrial5 scope and source/config hashes required')
    manifest=read(ROOT/c['manifest']);settings=read(ROOT/c['settings'])
    if [s['id'] for s in settings['datasets']]!=c['required_datasets']:raise ValueError('All five industrial transfer datasets required')
    records=[]
    for spec in settings['datasets']:
        dataset=next(x for x in manifest['datasets'] if x['id']==spec['id']);r=audit_dataset(ROOT,spec,dataset);records.append(r)
        print(json.dumps(dict(dataset=r['dataset'],arrays=r['array_count'],full_raw_reconstruction_passed=True)),flush=True)
    target=Path(c['report_path']);target.parent.mkdir(parents=True,exist_ok=True)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration_sha256=sha(ROOT/args.config),passed=True,records=records,
        full_datasets=5,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        dataset_preparation_verification_not_benchmark_performance=True,new_training_updates=0,new_neural_forwards=0,
        original_files_and_archives_preserved=True,all_original_experiments_complete=False)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(target),full_datasets=5,array_payloads=sum(x['array_count'] for x in records))),flush=True)


if __name__=='__main__':main()
