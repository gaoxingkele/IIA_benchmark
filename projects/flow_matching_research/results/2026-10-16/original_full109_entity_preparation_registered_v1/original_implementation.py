"""Prepare all109raw entities and explicit label controls, preserving sources."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[2]
MODULE='scripts.flow_matching.prepare_original_full_entity_tsad_v1'
CONFIG='configs/reproducibility/fm_original_full109_entity_preparation.2026-10-16_v1.json'
TEST='tests/test_fm_original_full109_entity_preparation_v1.py'
INVENTORY='configs/datasets/fm_original_entity_coverage_inventory.2026-10-16_v1.json'
REPORT='full109_preparation_and_mapping_audit.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def registration():
    inventory=read(ROOT/INVENTORY);proof_path=inventory['report_root']+'/actual_original_entity_coverage.json'
    proof=read(ROOT/proof_path)
    if not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/INVENTORY):raise ValueError('Actual full raw entity inventory required')
    groups=defaultdict(list)
    for row in proof['original_label_rows']:groups[(row['metadata']['spacecraft'],row['metadata']['chan_id'])].append(row)
    jobs=[]
    for dataset in ['SMAP','MSL']:
        for (ds,entity),rows in sorted(groups.items()):
            if ds!=dataset:continue
            jobs.append(dict(dataset=ds,entity=entity,loader='npy',raw={role:rows[0]['raw'][role] for role in ['train','test']},
                metadata_rows=[dict(source_row_index=r['source_row_index'],**r['metadata']) for r in rows]))
    smd=defaultdict(dict)
    for path,facts in proof['raw_files'].items():
        p=Path(path)
        if '/SMD/' in path:smd[p.stem][p.parent.name]=facts
    for entity,roles in sorted(smd.items()):
        if set(roles)!=set(['train','test','labels']):raise ValueError('All raw SMD roles required')
        jobs.append(dict(dataset='SMD',entity=entity,loader='text',raw=roles,metadata_rows=[]))
    if len(jobs)!=109:raise ValueError('54SMAP/27MSL/28SMD complete distinct entities required')
    guard_path='configs/reproducibility/fm_moment_ICM_all322_finite_control_closure.2026-10-15_v1.json'
    guard=read(ROOT/guard_path);recipe=read(ROOT/'configs/experiments/fm_tsad_execution.v1.json')['recipe']
    recipe=dict(recipe,minimum_validation_points=100,
        short_training_series='cut=min(floor(.8*N),N-100);fit_end=cut-100;all fit and validation lengths>=100',
        scaler='source constant_fit_v2 exact fit constants;fit segment mean/std for varying channels')
    pins=[INVENTORY,proof_path,guard_path,'configs/experiments/fm_tsad_execution.v1.json',MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/prepare_tsad_execution.py','scripts/flow_matching/prepare_tsad_constant_fit_v2.py',
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py','scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py',inventory['nasa_metadata']]
    return dict(schema_version=1,paper_citations=inventory['source_citations'],source_inventory=proof_path,jobs=jobs,
        required_entities=dict(SMAP=54,MSL=27,SMD=28),total_entities=109,recipe=recipe,
        label_controls=dict(SMAP=['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive'],
            MSL=['source_first_exclusive','source_first_inclusive'],SMD=['source_point_labels']),
        python=sys.executable,environment=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),pandas=importlib.metadata.version('pandas')),
        original_data_resource_policy=guard_path,settings=guard['settings'],resource_lock=guard['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(pins))],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full109_entity_tsad_v1',
        state_root='experiments/runs/fm_original_full109_entity_preparation_v1',
        raw_NASA_TEST_minmax_prescaling_disclosed=True,globally_inductive_source_data_certified=False,
        source_first_row_end_exclusive_primary=True,model_predictions_identical_across_test_label_controls_required=True,
        boundary='Full109distinct entities;no dropping short entities,source overwrite or existing queue changes. '
            'New raw complete-entity control uses minimum100validation points for D-12/T-9 before fitting. '
            'Every raw-to-output coordinate checked,all191CSV input coordinates compared to both original roles '
            'under explicit zero/one-based timestamps;no matching or label policy selected by model performance. '
            'Labels-only overlays do not require different model training or duplicated entity bootstrap groups. '
            'Existing45/681/273plus702and all other experiment scopes remain.')


def verify(config):
    if config!=registration():raise ValueError('Full109entities/all label controls/original data resource policy required')
    return config['jobs']


def label_array(rows,length,policy):
    import numpy as np
    if policy not in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']:
        raise ValueError('Declared first/last row and endpoint control required')
    row=rows[-1] if 'last' in policy else rows[0]
    if int(row['num_values'])!=length:raise ValueError('Exact full NASA label length required')
    labels=np.zeros(length,dtype=np.uint8)
    for a,b in ast.literal_eval(row['anomaly_sequences']):
        if not 0<=a<=b<length:raise ValueError('Original interval outside full test series')
        labels[a:b+int(policy.endswith('inclusive'))]=1
    return labels


def load_raw(job):
    import numpy as np
    if job['loader']=='npy':
        raw={role:np.load(facts['path'],mmap_mode='r',allow_pickle=False) for role,facts in job['raw'].items()}
    else:raw={role:np.loadtxt(facts['path'],delimiter=',',dtype=np.float64) for role,facts in job['raw'].items()}
    for role in ['train','test']:
        if raw[role].ndim!=2 or len(raw[role])<100:raise ValueError('Complete original multichannel series required')
    if raw['train'].shape[1]!=raw['test'].shape[1]:raise ValueError('All train/test channels must align')
    if 'labels' in raw:
        if raw['labels'].shape!=(len(raw['test']),) or not np.isin(raw['labels'],[0,1]).all():raise ValueError('Complete binary SMD target labels required')
        raw['labels']=raw['labels'].astype(np.uint8)
    return raw


def normalizer():
    """Original constant-fit body;one explicit new full-entity split rule."""
    import numpy as np
    from scripts.flow_matching import prepare_tsad_constant_fit_v2 as source
    module=source.constant_scaling_ast(Path(source.original.__file__).read_text(encoding='utf-8'))
    function=module.body[0];changed=0
    for node in function.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cut' for t in node.targets):
            node.value=ast.parse("min(int(len(train)*recipe['train_fraction']),len(train)-recipe['minimum_validation_points'])",mode='eval').body;changed+=1
    if changed!=1:raise ValueError('One declared validation-length boundary adjustment required')
    namespace=dict(np=np);exec(compile(ast.fix_missing_locations(module),'<full-original-entity-fit-normalization>','exec'),namespace)
    return namespace['split_and_normalize']


def independent_arrays(raw,record,arrays):
    """Rebuild every saved input from raw roles and independently checked stats."""
    import numpy as np
    start,end=record['fit_interval'];cut=record['validation_interval'][0]
    fit=np.asarray(raw['train'][start:end],dtype=np.float64).copy()
    median=np.array([np.median(column[np.isfinite(column)]) for column in fit.T])
    fit=np.where(np.isfinite(fit),fit,median)
    constant=np.max(fit,axis=0)==np.min(fit,axis=0)
    mean=fit.mean(axis=0);mean[constant]=fit[0,constant]
    std=fit.std(axis=0);std[constant]=0
    if not np.array_equal(median,record['median']) or not np.array_equal(mean,record['mean']) or not np.array_equal(std,record['std']):
        raise ValueError('Independent fit-only repair/scaling statistics must agree exactly')
    coordinates=0
    for role,values in [('train',raw['train'][:end]),('validation',raw['train'][cut:]),('test',raw['test'])]:
        values=np.asarray(values,dtype=np.float64);clean=np.where(np.isfinite(values),values,median)
        expected=((clean-mean)/np.where(std>0,std,1.)).astype(np.float32)
        if not np.array_equal(expected,arrays[role]) or not np.isfinite(arrays[role]).all():
            raise ValueError('Every original-source normalized coordinate must agree')
        coordinates+=expected.size
    return dict(every_normalized_coordinate_rebuilt=True,input_coordinates=coordinates,
        fit_only_statistics_independently_rebuilt=True,shared_NumPy_reader_and_reduction_kernel=True)


def compare_registered_CSV(job,raw,inventory):
    import numpy as np
    import pandas as pd
    reports=[]
    for item in inventory['registered_CSV_files']:
        if item['dataset']!=job['dataset'] or item['entity']!=job['entity']:continue
        if sha(item['path'])!=item['sha256']:raise ValueError('Original registered CSV changed')
        frame=pd.read_csv(item['path'],float_precision='round_trip')
        features=[c for c in frame.columns if c.startswith('feature_')]
        if len(features)!=raw['test'].shape[1] or features!=['feature_'+str(i+1) for i in range(len(features))]:
            raise ValueError('Complete registered feature order required')
        values=frame[features].to_numpy(np.float64);timestamps=frame['timestamp'].to_numpy(np.float64)
        if not np.isfinite(timestamps).all() or not np.equal(timestamps,np.floor(timestamps)).all() or len(set(timestamps))!=len(timestamps):
            raise ValueError('Exact unique integer timestamp mapping required')
        candidates=[]
        for raw_role in ['train','test']:
            original=np.asarray(raw[raw_role],dtype=np.float64)
            for origin in [0,1]:
                indices=timestamps.astype(np.int64)-origin
                if len(indices)==0 or indices.min()<0 or indices.max()>=len(original):continue
                expected=original[indices]
                equal=np.equal(values,expected)|(np.isnan(values)&np.isnan(expected))
                finite=np.isfinite(values)&np.isfinite(expected)
                maximum=float(np.max(np.abs(values[finite]-expected[finite]))) if finite.any() else None
                matched=bool(equal.all());label_matches={}
                if raw_role=='test':
                    if job['loader']=='npy':
                        for policy in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']:
                            label_matches[policy]=bool(np.array_equal(frame['is_anomaly'].to_numpy(),label_array(job['metadata_rows'],len(original),policy)[indices]))
                    else:label_matches['source_point_labels']=bool(np.array_equal(frame['is_anomaly'].to_numpy(),raw['labels'][indices]))
                candidates.append(dict(raw_role=raw_role,timestamp_origin=origin,coordinates=int(values.size),equal_coordinates=int(equal.sum()),
                    every_coordinate_equal=matched,maximum_finite_absolute_difference=maximum,
                    raw_row_start=int(indices.min()),raw_row_end_inclusive=int(indices.max()),label_policy_full_match=label_matches))
        reports.append(dict(path=item['path'],role=item['role'],points=len(frame),features=len(features),
            all_original_CSV_coordinates_compared=bool(candidates),CSV_float_reader='pandas_round_trip',
            timestamp_mapping_without_undocumented_offset_available=bool(candidates),
            CSV_labelled_anomaly_points=int(frame['is_anomaly'].sum()),candidates=candidates,
            exact_matching_candidates=[c for c in candidates if c['every_coordinate_equal']],
            cross_source_equivalence_selected_or_assumed=False))
    return reports


def worker(config):
    actual=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),pandas=importlib.metadata.version('pandas'))
    if actual!=config['environment']:raise ValueError('Frozen full-entity preprocessing environment required')
    import numpy as np
    inventory=read(ROOT/config['source_inventory']);normalize=normalizer()
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    manifests=[];cases=[];all_mapping=[]
    for dataset in ['SMAP','MSL','SMD']:
        payload={};overlays={p:{} for p in config['label_controls'][dataset]};records=[];source_files={}
        if dataset!='SMD':
            meta=inventory['original_label_metadata'];source_files[meta['path']]=dict(meta)
        for index,job in enumerate(j for j in config['jobs'] if j['dataset']==dataset):
            for role,facts in job['raw'].items():
                if sha(facts['path'])!=facts['sha256']:raise ValueError('Immutable full original raw file changed')
                source_files[facts['path']]=dict(path=facts['path'],sha256=facts['sha256'],bytes=facts['bytes'])
            raw=load_raw(job);primary=config['label_controls'][dataset][0]
            labels=label_array(job['metadata_rows'],len(raw['test']),primary) if job['loader']=='npy' else raw['labels']
            arrays,targets,record=normalize(raw['train'],raw['test'],labels,config['recipe'])
            independent=independent_arrays(raw,record,arrays)
            prefix=f'e{index:03d}';entity_id=dataset+'_'+job['entity']
            payload.update({prefix+'_'+role:value for role,value in arrays.items()});payload[prefix+'_labels']=targets
            for policy in overlays:
                overlays[policy][prefix+'_labels']=label_array(job['metadata_rows'],len(raw['test']),policy) if job['loader']=='npy' else raw['labels']
            mapping=compare_registered_CSV(job,raw,inventory);all_mapping.extend([dict(dataset=dataset,entity=job['entity'],**r) for r in mapping])
            source_cut=int(len(raw['train'])*config['recipe']['train_fraction']);cut=record['validation_interval'][0]
            metadata=dict(entity_id=entity_id,array_prefix=prefix,**record,raw_entity=job['entity'],
                source_training_80percent_cut=source_cut,short_series_validation_adjustment=cut!=source_cut,
                numerical_audit=independent,duplicate_label_source_rows=job['metadata_rows'],
                bootstrap_group=entity_id,raw_NASA_already_TEST_minmax_prescaled=job['loader']=='npy')
            records.append(metadata)
            directory=output/'entities'/entity_id;directory.mkdir(parents=True,exist_ok=False)
            for role,bounds in [('train',record['fit_interval']),('validation',record['validation_interval']),('test',[0,len(raw['test'])])]:
                with (directory/(role+'_original_rows.npy')).open('xb') as stream:np.save(stream,np.arange(*bounds,dtype=np.int64),allow_pickle=False)
            write(directory/'preprocessing.json',dict(job=job,record=metadata,mapping=mapping,source_raw_files_unchanged=True))
            for facts in job['raw'].values():
                if sha(facts['path'])!=facts['sha256']:raise ValueError('Raw file changed during full preparation')
            cases.append(dict(dataset=dataset,entity=job['entity'],short_series_adjustment=cut!=source_cut,**independent))
        target=output/(dataset.lower()+'.npz')
        with target.open('xb') as stream:np.savez_compressed(stream,**payload)
        with np.load(target,allow_pickle=False) as saved:
            if set(saved.files)!=set(payload) or any(not np.array_equal(saved[k],v) for k,v in payload.items()):raise ValueError('Every serialized input coordinate and target must agree')
        controls={}
        for policy,labels in overlays.items():
            path=output/'label_controls'/(dataset+'__'+policy+'.npz');path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:np.savez_compressed(stream,**labels)
            with np.load(path,allow_pickle=False) as saved:
                if set(saved.files)!=set(labels) or any(not np.array_equal(saved[k],v) for k,v in labels.items()):raise ValueError('All complete test-label overlays must agree')
            controls[policy]=dict(path=str(path),sha256=sha(path),test_anomaly_points=sum(int(v.sum()) for v in labels.values()),
                uses_identical_training_validation_inputs_and_model_predictions=True)
        manifests.append(dict(id=dataset.lower(),input_npz=str(target),input_sha256=sha(target),entities=records,
            source_files=list(source_files.values()),grouped_splits_verified=True,labels_only_controls=controls,
            primary_label_policy=config['label_controls'][dataset][0],source_variant='original_full_entity_fit_only_v1',
            historical_author_equivalence_certified=False,global_source_inductivity_certified=False))
    if len(cases)!=109 or len(all_mapping)!=191:raise ValueError('All109raw entities and all191original registered CSVs required')
    verify(config)
    manifest_path=output/'manifest.json';write(manifest_path,dict(datasets=manifests,recipe=config['recipe'],configuration_sha256=sha(ROOT/CONFIG),
        label_controls_selectable_by_test_performance=False,original_raw_data_preserved=True))
    write(output/REPORT,dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),environment=actual,entities=109,datasets=3,
        required_entity_counts=config['required_entities'],cases=cases,CSV_mapping=all_mapping,CSV_files_compared=191,
        full_input_coordinate_reconstruction=True,complete_serialized_arrays_verified=True,
        prepared_manifest=str(manifest_path),prepared_manifest_sha256=sha(manifest_path),
        raw_NASA_TEST_minmax_prescaling_disclosed=True,model_training_or_inference_performed=False,
        full_raw_dataset_model_experiments_complete=False,all_original_experiments_complete=False))


def controller(config):
    from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard
    node=next(n for n in ast.parse(Path(guard.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    namespace=dict(guard.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    replacements={'closing_all1949_anomaly_sources':'preparing_all109_original_multivariate_entities',
        'all1949_anomaly_cache_closure_passed':'all109_original_entity_preparation_passed'}
    class StateNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=StateNames().visit(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(guard.__file__),'exec'),namespace)
    return namespace['controller'](config)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    if args.register:
        write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,entities=109,datasets=3)));return
    config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,entities=109,CSV_files=191,label_controls=7,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,actual_model_experiments_completed=0)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
