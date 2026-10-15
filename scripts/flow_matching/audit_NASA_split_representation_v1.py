"""Characterize all fixed input serialization hypotheses for G-7 and T-13.

No numerical tolerance,score/model,label-policy selection or raw rewriting.
Matching a declared representation does not certify historical generator code.
"""
from __future__ import annotations
import argparse
import ast
import io
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as data

ROOT,read,sha=data.ROOT,data.read,data.sha
MODULE='scripts.flow_matching.audit_NASA_split_representation_v1'
CONFIG='configs/reproducibility/fm_NASA_two_split_serialization_audit.2026-10-16_v1.json'
TEST='tests/test_fm_NASA_split_representation_v1.py'
REPORT='two_split_serialization_audit.json'
PROFILES=('direct_float64','CSV_pandas_default_C_read','CSV_pandas_round_trip_read','float32_then_float64')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def registration():
    source=read(ROOT/data.CONFIG);inventory=read(ROOT/source['source_inventory'])
    settings_path='configs/reproducibility/fm_timesnet_all5_NASA_label_protocol.2026-10-16_v2.json'
    settings=read(ROOT/settings_path);base=read(ROOT/data.INVENTORY)['registered_CSV_root']
    jobs=[]
    for dataset,entity in [('SMAP','G-7'),('MSL','T-13')]:
        job=next(j for j in source['jobs'] if j['dataset']==dataset and j['entity']==entity)
        csv={role:dict(path=str(ROOT/base/dataset/(dataset+'_'+entity+'_'+role+'.csv')),
            sha256=sha(ROOT/base/dataset/(dataset+'_'+entity+'_'+role+'.csv'))) for role in ['train','val','test']}
        jobs.append(dict(raw_job=job,CSV_roles=csv))
    pins=[data.CONFIG,source['source_inventory'],settings_path,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/prepare_original_full_entity_tsad_v1.py',
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py']
    return dict(schema_version=1,jobs=jobs,required_entities=2,required_CSV_files=6,
        profiles=list(PROFILES),raw_roles=['train','test'],timestamp_origins=[0,1],
        all_declared_comparisons=96,model_scores_used=False,labels_used_to_select_representation=False,
        numpy_tolerance_used=False,old_strict_float64_mapping_gate_changed=False,
        environment=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),pandas=importlib.metadata.version('pandas')),
        python=sys.executable,settings=settings['settings'],resource_lock=settings['resource_lock'],
        state_root='experiments/runs/fm_NASA_two_split_serialization_audit_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/NASA_two_split_serialization_audit_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        citations=['https://github.com/PLAN-Lab/mTSBench','https://github.com/khundman/telemanom'],
        boundary='All6CSV roles/all4fixed representations/all2raw roles/all2timestamp origins. '
            'Profile matching is exact equality under a named parameter-free transform,not a relaxed tolerance. '
            'Record every match/failure;never choose by any model performance. Current software transformation '
            'compatibility does not certify unknown historical writer/version. Source files and old model/mapping gates unchanged. '
            'Full109rawdata/873experiments and all other paper obligations remain.')


def verify(config):
    if config!=registration():raise ValueError('All6files/96declared comparisons and unchanged exact-only policy required')
    return config['jobs']


def representation(raw,profile):
    import numpy as np
    import pandas as pd
    raw=np.asarray(raw,dtype=np.float64)
    if profile=='direct_float64':return raw.copy()
    if profile=='float32_then_float64':return raw.astype(np.float32).astype(np.float64)
    if profile not in ['CSV_pandas_default_C_read','CSV_pandas_round_trip_read']:raise ValueError('Only fixed preregistered serialization profiles')
    stream=io.StringIO();pd.DataFrame(raw).to_csv(stream,index=False,header=False);stream.seek(0)
    kwargs=dict(header=None,engine='c')
    if profile=='CSV_pandas_round_trip_read':kwargs['float_precision']='round_trip'
    return pd.read_csv(stream,**kwargs).to_numpy(np.float64)


def compare(values,timestamps,candidate,origin):
    import numpy as np
    if (values.ndim!=2 or candidate.ndim!=2 or candidate.shape[1]!=values.shape[1]
            or len(values)!=len(timestamps) or not len(values) or not np.isfinite(timestamps).all()
            or not np.equal(timestamps,np.floor(timestamps)).all() or np.any(np.diff(timestamps)<=0)):
        raise ValueError('Exact complete features and monotonic unique integer timestamps required')
    rows=timestamps.astype(np.int64)-origin
    if rows.min()<0 or rows.max()>=len(candidate):
        return dict(in_bounds=False,every_coordinate_equal=False,compared_coordinates=0),None
    expected=candidate[rows];same=np.equal(values,expected)|(np.isnan(values)&np.isnan(expected))
    finite=np.isfinite(values)&np.isfinite(expected)
    return dict(in_bounds=True,every_coordinate_equal=bool(same.all()),compared_coordinates=int(values.size),
        unequal_coordinates=int((~same).sum()),maximum_finite_absolute_difference=float(np.max(np.abs(values[finite]-expected[finite]))) if finite.any() else None,
        first_original_row=int(rows[0]),last_original_row=int(rows[-1]),
        original_rows_contiguous=bool(np.array_equal(rows,np.arange(rows[0],rows[0]+len(rows))))),rows


def partition(val_rows,test_rows,total):
    import numpy as np
    union=np.concatenate([val_rows,test_rows]);unique=np.unique(union)
    return dict(validation_rows=len(val_rows),test_rows=len(test_rows),overlap_rows=len(union)-len(unique),
        every_original_test_row_once=bool(len(union)==total and np.array_equal(unique,np.arange(total))),
        validation_precedes_test=bool(len(val_rows) and len(test_rows) and val_rows.max()<test_rows.min()))


def worker(config):
    actual=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),pandas=importlib.metadata.version('pandas'))
    if actual!=config['environment']:raise ValueError('Frozen serialization environment required')
    import numpy as np
    import pandas as pd
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    results=[];comparisons=0
    for job in config['jobs']:
        original=job['raw_job'];name=original['dataset']+'_'+original['entity'];folder=output/name;folder.mkdir()
        raw={};transforms={};files={}
        for role,facts in original['raw'].items():
            if sha(facts['path'])!=facts['sha256']:raise ValueError('Full immutable raw source changed')
            raw[role]=np.load(facts['path'],mmap_mode='r',allow_pickle=False)
            for profile in config['profiles']:
                value=representation(raw[role],profile)
                if value.shape!=raw[role].shape:raise ValueError('Serialization must preserve every source row/channel')
                path=folder/(role+'__'+profile+'.npy')
                with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
                with path.open('rb') as stream:
                    if not np.array_equal(np.load(stream,allow_pickle=False),value,equal_nan=True):raise ValueError('Full serialized representation must be exact')
                transforms[role,profile]=value;files[role+'__'+profile]=dict(path=str(path),sha256=sha(path),shape=list(value.shape))
        CSVs={};index_matches={}
        for csv_role,facts in job['CSV_roles'].items():
            if sha(facts['path'])!=facts['sha256']:raise ValueError('Registered input CSV changed')
            frame=pd.read_csv(facts['path'],float_precision='round_trip')
            features=[c for c in frame if c.startswith('feature_')]
            if features!=['feature_'+str(i+1) for i in range(raw['test'].shape[1])]:raise ValueError('All input feature identities required')
            values=frame[features].to_numpy(np.float64);times=frame['timestamp'].to_numpy(np.float64);candidates=[]
            for raw_role in config['raw_roles']:
                for origin in config['timestamp_origins']:
                    for profile in config['profiles']:
                        evidence,rows=compare(values,times,transforms[raw_role,profile],origin);comparisons+=1
                        if evidence['every_coordinate_equal']:
                            index_matches[csv_role,raw_role,origin,profile]=rows
                        label_matches={}
                        if evidence['every_coordinate_equal'] and raw_role=='test' and 'is_anomaly' in frame:
                            for label_policy in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']:
                                labels=data.label_array(original['metadata_rows'],len(raw['test']),label_policy)[rows]
                                label_matches[label_policy]=bool(np.array_equal(labels,frame['is_anomaly'].to_numpy()))
                        candidates.append(dict(raw_role=raw_role,timestamp_origin=origin,profile=profile,**evidence,label_policy_exact_matches=label_matches))
            CSVs[csv_role]=dict(path=facts['path'],sha256=facts['sha256'],rows=len(frame),features=len(features),
                timestamp_first=float(times[0]),timestamp_last=float(times[-1]),provided_labels='is_anomaly' in frame,
                provided_anomaly_points=int(frame['is_anomaly'].sum()) if 'is_anomaly' in frame else None,candidates=candidates)
        partitions=[]
        for origin in config['timestamp_origins']:
            for profile in config['profiles']:
                v=index_matches.get(('val','test',origin,profile));t=index_matches.get(('test','test',origin,profile))
                if v is not None and t is not None:
                    partitions.append(dict(profile=profile,timestamp_origin=origin,**partition(v,t,len(raw['test']))))
        for facts in [*original['raw'].values(),*job['CSV_roles'].values()]:
            if sha(facts['path'])!=facts['sha256']:raise ValueError('Any source file changed during readonly audit')
        result=dict(entity=name,raw_source=original,representation_files=files,CSV_roles=CSVs,complete_raw_TEST_partitions=partitions,
            historical_generator_or_version_certified=False,numerical_tolerance_used=False,
            model_scores_or_metrics_used_for_profile_selection=False,original_mapping_gate_unchanged=True)
        write(folder/'representation_audit.json',result);results.append(result)
    if comparisons!=96:raise ValueError('Every preregistered source/role/origin/profile comparison required')
    verify(config)
    write(output/REPORT,dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),environment=actual,
        entities=2,CSV_files=6,declared_comparisons_executed=comparisons,results=results,
        source_files_modified=False,original_model_or_label_audits_retried=False,model_training_or_inference_performed=False,
        historical_generator_certified=False,all_original_experiments_complete=False))


def controller(config):
    from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard
    node=next(n for n in ast.parse(Path(guard.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    replacements={'closing_all1949_anomaly_sources':'auditing_two_split_NASA_serialization_controls',
        'all1949_anomaly_cache_closure_passed':'two_split_NASA_serialization_audit_passed'}
    class StateNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=StateNames().visit(node);namespace=dict(guard.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(guard.__file__),'exec'),namespace)
    return namespace['controller'](config)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,declared_comparisons=96)));return
    config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,CSV_files=6,declared_comparisons=96,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,model_training_performed=False)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
