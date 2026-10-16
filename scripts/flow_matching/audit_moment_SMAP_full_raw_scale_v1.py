"""Independently reconstruct both registered SMAP preprocessing variants.

Read-only full raw CSV audit. It supplements the completed MOMENT native
replay, does not replace any broader data track, inference or queued method.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.audit_moment_SMAP_full_raw_scale_v1'
CONFIG='configs/reproducibility/fm_moment_SMAP_full_raw_scale_audit.v1.json'


def registration():
    settings='configs/experiments/fm_tsad_execution.v1.json'
    main='data/public_datasets/flow_matching/processed_tsad_execution_v1/manifest.json'
    control='data/public_datasets/flow_matching/processed_tsad_constant_fit_v2/manifest.json'
    paths=[settings,main,control,MODULE.replace('.','/')+'.py','tests/test_fm_moment_SMAP_full_raw_scale_v1.py']
    return dict(dataset='smap',preparation_settings=settings,original_manifest=main,constant_fit_manifest=control,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        expected_available_paired_entities=51,
        report_directory='F:/aicoding/IIA_Data/experiments/moment_SMAP_full_raw_scale_audit_v1',
        settings=dict(maximum_RSS_bytes=512*2**20,maximum_private_bytes=1024*2**20,minimum_commit_headroom_bytes=16*2**30),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        new_training_or_inference_jobs=0,broader_data_tracks_and_experiments_preserved=True,
        citation='https://arxiv.org/abs/2402.03885',data_citation='https://github.com/decisionintelligence/TAB',
        all_original_experiments_complete=False)


def reconstructed(train,test,labels,recipe,constant_correction=False):
    """Independent literal fit-only statistics, never call the prepared-data code."""
    import numpy as np
    train=np.asarray(train,dtype=np.float64);test=np.asarray(test,dtype=np.float64)
    cut=int(len(train)*recipe['train_fraction']);fit_end=cut-recipe['purge_points']
    if min(fit_end,len(train)-cut,len(test))<recipe['minimum_segment_points']:raise ValueError('Full registered split required')
    arrays={'train':train[:fit_end].copy(),'validation':train[cut:].copy(),'test':test.copy()}
    fit=arrays['train'];median=np.nanmedian(np.where(np.isfinite(fit),fit,np.nan),axis=0)
    if not np.isfinite(median).all():raise ValueError('Every fit channel needs finite observations')
    for values in arrays.values():
        invalid=~np.isfinite(values)
        values[invalid]=np.broadcast_to(median,values.shape)[invalid]
    mean=arrays['train'].mean(axis=0);std=arrays['train'].std(axis=0)
    exact_constant=np.all(arrays['train']==arrays['train'][0:1],axis=0)
    legacy_mean=mean.copy();legacy_std=std.copy()
    if constant_correction:
        mean[exact_constant]=arrays['train'][0,exact_constant];std[exact_constant]=0.
    denominator=np.where(std==0.,1.,std)
    scaled={role:((values-mean)/denominator).astype(np.float32) for role,values in arrays.items()}
    label_array=np.asarray(labels)
    if label_array.shape!=(len(test),) or not np.isin(label_array,[0,1]).all():raise ValueError('Complete raw binary test labels required')
    return scaled,label_array.astype(np.uint8),dict(mean=mean,std=std,median=median,
        legacy_mean=legacy_mean,legacy_std=legacy_std,exact_constant=exact_constant,
        fit_interval=[0,fit_end],purged_interval=[fit_end,cut],validation_interval=[cut,len(train)])


def require_bytes(actual,expected):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
        raise ValueError('Every full raw-derived value must match the frozen archive byte-for-byte')


def run(c):
    import numpy as np
    import pandas as pd
    import psutil
    from scripts.flow_matching.heavy_resources import resource_snapshot
    import platform,time
    if c!=registration():raise ValueError('Full51 raw source/split/scale contract required')
    for pin in c['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen audit source changed')
    original=read(ROOT/c['original_manifest']);control=read(ROOT/c['constant_fit_manifest'])
    a=next(d for d in original['datasets'] if d['id']==c['dataset']);b=next(d for d in control['datasets'] if d['id']==c['dataset'])
    prep=read(ROOT/c['preparation_settings']);spec=next(s for s in prep['datasets'] if s['id']==c['dataset'])
    if a['source_files']!=b['source_files']:
        raise ValueError('Every original available raw pair must remain')
    if len(a['entities'])!=c['expected_available_paired_entities'] or len(b['entities'])!=len(a['entities']):raise ValueError('Full51 pairs required')
    for d in [a,b]:
        if sha(ROOT/d['input_npz'])!=d['input_sha256']:raise ValueError('Full frozen prepared archive changed')
    for source in a['source_files']:
        if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Original raw CSV changed')
    target=Path(c['report_directory']);target.mkdir(parents=True,exist_ok=False)
    records=[];peak_rss=peak_private=0;start=time.monotonic()
    with np.load(ROOT/a['input_npz'],allow_pickle=False) as old,np.load(ROOT/b['input_npz'],allow_pickle=False) as new:
        if set(old.files)!=set(new.files):raise ValueError('Every original array must remain')
        for entity,corrected in zip(a['entities'],b['entities']):
            memory=psutil.Process().memory_info();peak_rss=max(peak_rss,memory.rss);peak_private=max(peak_private,getattr(memory,'private',memory.vms))
            if (peak_rss>c['settings']['maximum_RSS_bytes'] or peak_private>c['settings']['maximum_private_bytes']
                    or resource_snapshot()['commit_available_bytes']<c['settings']['minimum_commit_headroom_bytes']):
                raise RuntimeError('Read-only audit budget reached;preserve prefix evidence')
            identity=entity['entity_id'];prefix=entity['array_prefix']
            if corrected['entity_id']!=identity or corrected['array_prefix']!=prefix:raise ValueError('Raw entity order changed')
            root=ROOT/spec['root'];frames={role:pd.read_csv(root/(identity+'_'+role+'.csv')) for role in ['train','test']}
            excluded=set(spec['excluded_columns'])|{spec['label_column']}
            features=[k for k in frames['train'].columns if k not in excluded]
            if (features!=entity['feature_names'] or features!=[k for k in frames['test'].columns if k not in excluded]
                    or len(features)!=entity['features']):raise ValueError('All raw feature columns/order required')
            for frame in frames.values():
                if frame['timestamp'].duplicated().any() or not frame['timestamp'].is_monotonic_increasing:raise ValueError('Raw timestamp order changed')
            train=frames['train'][features].to_numpy(dtype=np.float64);test=frames['test'][features].to_numpy(dtype=np.float64)
            labels=frames['test'][spec['label_column']].to_numpy()
            modes={}
            for fixed,manifest,archive,name in [(False,entity,old,'original'),(True,corrected,new,'constant_fit_v2')]:
                arrays,truth,stats=reconstructed(train,test,labels,original['recipe'],fixed)
                for key in ['fit_interval','purged_interval','validation_interval']:
                    if stats[key]!=manifest[key]:raise ValueError('Complete fit/purge/validation boundaries differ')
                for key in ['mean','std','median']:
                    if not np.array_equal(stats[key],np.asarray(manifest[key],dtype=np.float64)):
                        raise ValueError('Independent full fit statistic differs: '+identity+'/'+name+'/'+key)
                for role,value in arrays.items():require_bytes(value,archive[prefix+'_'+role])
                require_bytes(truth,archive[prefix+'_labels'])
                modes[name]=dict(full_values_and_labels_bitwise_equal=True,
                    max_absolute_input_by_role={role:float(np.max(np.abs(value))) for role,value in arrays.items()},
                    points_by_role={role:len(value) for role,value in arrays.items()},
                    positive_test_labels=int(truth.sum()))
            affected=np.flatnonzero(stats['exact_constant']&(stats['legacy_std']>0)).tolist()
            varying=np.flatnonzero(~stats['exact_constant'])
            for role in ['train','validation','test']:
                require_bytes(old[prefix+'_'+role][:,varying],new[prefix+'_'+role][:,varying])
            row=dict(entity=identity,full_channels=len(features),variants=modes,
                strict_constant_but_legacy_nonzero_std_channels=affected,
                legacy_std_on_affected_channels=[float(stats['legacy_std'][i]) for i in affected],
                nonconstant_channel_values_identical=True)
            records.append(row);exclusive(target/'entities'/(identity+'.json'),row)
    for d in [a,b]:
        if sha(ROOT/d['input_npz'])!=d['input_sha256']:raise ValueError('Input archive modified during audit')
    for source in a['source_files']:
        if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Raw CSV modified during audit')
    report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration_sha256=sha(ROOT/CONFIG),
        full_registered_available_raw_pairs_verified=len(records),full_registered_preprocessing_variants_verified=2,
        original_input_sha256=a['input_sha256'],constant_fit_input_sha256=b['input_sha256'],
        full_original_raw_CSV_hashes_verified=len(a['source_files']),all_raw_columns_splits_labels_and_scaled_values_reconstructed=True,
        changed_constant_fit_entities=[r['entity'] for r in records if r['strict_constant_but_legacy_nonzero_std_channels']],
        global_max_absolute_inputs={name:{role:max(r['variants'][name]['max_absolute_input_by_role'][role] for r in records) for role in ['train','validation','test']} for name in ['original','constant_fit_v2']},
        native_points_by_role={role:sum(r['variants']['original']['points_by_role'][role] for r in records) for role in ['train','validation','test']},
        test_positive_labels=sum(r['variants']['original']['positive_test_labels'] for r in records),
        entities=records,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        peak_RSS_bytes=peak_rss,peak_private_bytes=peak_private,elapsed_seconds=time.monotonic()-start,
        native_inference_executed=False,new_completed_model_slots=0,original_data_modified=False,
        scope='All51 pairs in the exact registered local SMAP manifest;broader data tracks and unfinished experiments remain required',
        original_scope=c['original_scope'],all_original_experiments_complete=False)
    exclusive(target/'full_raw_preprocessing_reconstruction.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['passed','full_registered_available_raw_pairs_verified','changed_constant_fit_entities','global_max_absolute_inputs','native_points_by_role','test_positive_labels','elapsed_seconds']},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    run(read(ROOT/CONFIG))


if __name__=='__main__':main()
