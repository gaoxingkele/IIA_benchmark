"""Run all seven existing local TSAD families on the full original raw entities.

Only the data binding/output identity changes. Every native numerical worker,
model configuration, seed, capacity, precision, epoch and phase gate remains.
Author-specific experiments and independent replay/statistics remain separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as preparation

ROOT,read,sha=preparation.ROOT,preparation.read,preparation.sha
MODULE='scripts.flow_matching.run_original_full_raw873_experiments_v1'
CONFIG='configs/reproducibility/fm_original_full_raw873_experiments.2026-10-16_v1.json'
TEST='tests/test_fm_original_full_raw873_experiments_v1.py'
CPU_POLICY='configs/runtime/fm_tsad_constant_fit_cpu_controls.v2.json'
GPU_POLICY='configs/runtime/fm_tsad_constant_fit_gpu_native_resource.v2.json'
FAMILIES={
    'objectives':('configs/experiments/fm_tsad_cpu_queue.v1.json',120),
    'neural':('configs/experiments/fm_tsad_gpu_queue.v1.json',510),
    'baselines':('configs/experiments/fm_strict_baseline_queue.v1.json',105),
    'entropic':('configs/experiments/fm_tsad_entropic_repair_queue.v1.json',45),
    'reflow':('configs/experiments/fm_reflow_queue.v1.json',60),
    'tranad64':('configs/experiments/fm_tranad_reference_float64_queue.v1.json',15),
    'moment':('configs/experiments/fm_moment_pretrained_queue.v1.json',18)}
DATASETS=('smap','msl','smd')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def update_file(path,value):
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic
    atomic(path,value)


def selected(queue):return [j for j in queue['jobs'] if j['dataset'] in DATASETS]


def registration():
    from scripts.flow_matching.run_light_controller import FUNCTIONS
    source_data=read(ROOT/preparation.CONFIG);preparation.verify(source_data)
    cpu,gpu=read(ROOT/CPU_POLICY),read(ROOT/GPU_POLICY)
    records=[];ids=[];pins={preparation.CONFIG,CPU_POLICY,GPU_POLICY,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/prepare_original_full_entity_tsad_v1.py',
        'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'scripts/flow_matching/run_bounded_strict_cpu_queue_v2.py',
        'scripts/flow_matching/run_tsad_native_gpu_resource_queue_v1.py',
        'scripts/flow_matching/run_spectral_original_queue.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_moment_pretrained_queue.py','scripts/flow_matching/run_moment_pretrained_job.py',
        'scripts/flow_matching/run_tsad_experiment.py'}
    pins.update(FUNCTIONS)
    for name,(path,count) in FAMILIES.items():
        q=read(ROOT/path);jobs=selected(q)
        if len(jobs)!=count or len({j['id'] for j in jobs})!=count:raise ValueError('Every original dataset/method/seed required')
        ids.extend(j['id'] for j in jobs);pins.add(path)
        pins.update(j['model_config'] for j in jobs)
        records.append(dict(name=name,source_queue=path,selected_source_ids=[j['id'] for j in jobs],required_cases=count,
            original_evaluation=q['evaluation'],native_worker_python=q['python'],
            resource_policy=GPU_POLICY if name=='neural' else CPU_POLICY,
            controller_source='scripts/flow_matching/run_moment_pretrained_queue.py' if name=='moment' else 'scripts/flow_matching/run_tsad_queue.py'))
    if len(ids)!=873 or len(set(ids))!=873:raise ValueError('All873distinct source jobs across all7families required')
    return dict(schema_version=1,full_entity_preparation=preparation.CONFIG,families=records,required_cases=873,
        required_cases_per_dataset=dict(smap=291,msl=291,smd=291),datasets=list(DATASETS),
        CPU_settings=cpu['settings'],CPU_resource_lock=cpu['resource_lock'],
        GPU_settings=gpu['settings'],GPU_resource_lock=gpu['resource_lock'],
        state_root='experiments/runs/fm_original_full_raw873_experiments_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full_raw873_experiments_v1',
        controller_python=sys.executable,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        test_label_controls='All7label versions use the same trained/published predictions;no test-based policy choice',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        boundary='All873existing source jobs on109full original entities,not a reduced method shortlist. '
            'Data-ready gate requires actual complete preparation and successful unchanged resource guard. '
            'Every source model config/window/seed/device/epoch/precision/evaluation/phase gate remains. '
            'All large outputs and materialized queues onF. No automatic retry. '
            'Source training/scoring success is not independent model replay,five-seed statistics,fullTAB range '
            'or author reproduction. All author-specific MaelNet/Pi/CrossAD,ICM pretraining and other papers remain.')


def verify(config):
    if config!=registration():raise ValueError('All873source cases and original numerical/resource policies must remain')
    return config['families']


def ready_data(config):
    source=read(ROOT/config['full_entity_preparation'])
    p=Path(source['artifact_root'])/preparation.REPORT;g=ROOT/source['state_root']/'guard_receipt.json'
    if g.exists():
        status=read(g)
        if not status['passed'] or status['exit_code']!=0 or status['resource_guard_aborted']:
            raise ValueError('Original full109data preparation failed;preserve and diagnose without retry')
        if not p.exists():raise ValueError('Successful original data guard has no complete source proof')
    if not p.exists() or not g.exists():return None
    proof,guard=read(p),read(g)
    expected_cases={(j['dataset'],j['entity']) for j in source['jobs']}
    if (not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/config['full_entity_preparation'])
            or proof['entities']!=109 or proof['datasets']!=3 or proof['CSV_files_compared']!=191
            or not proof['full_input_coordinate_reconstruction'] or not proof['complete_serialized_arrays_verified']
            or len(proof['cases'])!=109 or not all(r['every_normalized_coordinate_rebuilt'] for r in proof['cases'])
            or {(r['dataset'],r['entity']) for r in proof['cases']}!=expected_cases
            or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(p)):
        raise ValueError('Actual full109data and successful original preparation guard required before model execution')
    manifest=Path(proof['prepared_manifest'])
    if sha(manifest)!=proof['prepared_manifest_sha256']:raise ValueError('Actual full prepared manifest changed')
    data=read(manifest);records={r['id']:r for r in data['datasets']}
    if set(records)!=set(DATASETS):raise ValueError('All3full original datasets required')
    for name,record in records.items():
        if (len(record['entities'])!=source['required_entities'][name.upper()]
                or {e['entity_id'] for e in record['entities']}!={d+'_'+entity for d,entity in expected_cases if d==name.upper()}
                or not record['grouped_splits_verified'] or sha(ROOT/record['input_npz'])!=record['input_sha256']):
            raise ValueError('Complete original entity arrays and grouped splits required')
        controls=record['labels_only_controls']
        if set(controls)!=set(source['label_controls'][name.upper()]):raise ValueError('All declared labels-only controls required')
        for control in controls.values():
            if sha(control['path'])!=control['sha256'] or not control['uses_identical_training_validation_inputs_and_model_predictions']:
                raise ValueError('Immutable full test-label control files required')
    return dict(manifest=str(manifest),manifest_sha256=sha(manifest),records=records,
        full_data_proof_sha256=sha(p),full_data_guard_sha256=sha(g),
        every_numeric_coordinate_proven=True,full_NASA_source_inductivity_certified=False)


def clone_queue(config,family,binding):
    source=read(ROOT/family['source_queue']);queue=deepcopy(source);jobs=[]
    for old in selected(source):
        job=deepcopy(old);identity=old['id']+'__original_full109_v1'
        job.update(id=identity,dataset_manifest_path=binding['manifest'],
            output_directory=config['artifact_root']+'/'+family['name']+'/jobs/'+identity,
            full_raw_source_job_id=old['id'],full_raw_source_queue=family['source_queue'],
            full_raw_data_variant='original_full109_entity_fit_only_v1')
        pins={p['path']:p['sha256'] for p in old['frozen_files']}
        record=binding['records'][job['dataset']]
        pins.update({binding['manifest']:binding['manifest_sha256'],record['input_npz']:record['input_sha256'],
            CONFIG:sha(ROOT/CONFIG)})
        pins.update({r['path']:r['sha256'] for r in record['source_files']})
        job['frozen_files']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
        jobs.append(job)
    queue.update(jobs=jobs,state_root=config['artifact_root']+'/'+family['name']+'/source_controller',
        original_full_raw_source_queue=family['source_queue'],
        full_raw_data_manifest_sha256=binding['manifest_sha256'],
        boundary=source['boundary']+'; Additional full109raw entity experiment;all original fidelity limits remain.')
    validate_clone(source,queue,family,binding)
    return queue


def validate_clone(source,queue,family,binding):
    expected=selected(source);actual=queue['jobs']
    if len(actual)!=family['required_cases'] or len(expected)!=len(actual):raise ValueError('All source methods/seeds required')
    allowed={'id','dataset_manifest_path','output_directory','frozen_files','full_raw_source_job_id','full_raw_source_queue','full_raw_data_variant'}
    for old,new in zip(expected,actual):
        if (new['id']!=old['id']+'__original_full109_v1' or new['full_raw_source_job_id']!=old['id']
                or new['full_raw_source_queue']!=family['source_queue'] or new['full_raw_data_variant']!='original_full109_entity_fit_only_v1'
                or new['dataset_manifest_path']!=binding['manifest'] or new['output_directory']==old['output_directory']
                or {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in new.items() if k not in allowed}):
            raise ValueError('Only the explicit new data/output/provenance may differ')
        pins={p['path']:p['sha256'] for p in new['frozen_files']}
        if any(pins.get(p['path'])!=p['sha256'] for p in old['frozen_files']):raise ValueError('All original frozen receipts must remain')
        record=binding['records'][new['dataset']]
        required={binding['manifest']:binding['manifest_sha256'],record['input_npz']:record['input_sha256'],CONFIG:sha(ROOT/CONFIG)}
        required.update({p['path']:p['sha256'] for p in record['source_files']})
        if any(pins.get(p)!=h for p,h in required.items()):raise ValueError('Every new full raw array,source and registration must be frozen')
    for key,value in source.items():
        if key not in ['jobs','state_root','boundary'] and queue.get(key)!=value:
            raise ValueError('Original worker environment/evaluation/phase gates must remain')


def materialize(config,binding):
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    queues=[];ids=[];datasets=Counter()
    for family in config['families']:
        queue=clone_queue(config,family,binding);path=output/'queues'/(family['name']+'.json');write(path,queue)
        queues.append(dict(family=family['name'],path=str(path),sha256=sha(path),jobs=len(queue['jobs'])))
        ids.extend(j['id'] for j in queue['jobs']);datasets.update(j['dataset'] for j in queue['jobs'])
    if len(ids)!=873 or len(set(ids))!=873 or dict(datasets)!=config['required_cases_per_dataset']:
        raise ValueError('Full873jobs and291per full original dataset required')
    proof=dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),full_data_binding=binding,queues=queues,
        required_cases=873,case_ids=ids,required_cases_per_dataset=dict(datasets),
        trained_models_complete=False,independent_native_replay_complete=False,full_TAB_and_label_control_metrics_complete=False,
        all_original_experiments_complete=False)
    write(output/'materialization.json',proof);return proof


def verify_materialization(config):
    path=Path(config['artifact_root'])/'materialization.json'
    if not path.exists():raise ValueError('Actual full data and frozen873queues required before NN imports')
    proof=read(path);binding=ready_data(config)
    if (binding is None or not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or proof['full_data_binding']!=binding or proof['required_cases']!=873):
        raise ValueError('Actual full data and frozen873queues required before NN imports')
    for item in proof['queues']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Materialized scientific queue changed')
    return proof


def family_controller(config,name):
    import ast
    import psutil
    from scripts.flow_matching.run_light_controller import load_controller
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    from scripts.flow_matching import run_resource_gated_light_controller_v1 as cpu_guard
    from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import capped_class_ast
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    proof=verify_materialization(config);family=next(f for f in config['families'] if f['name']==name)
    item=next(r for r in proof['queues'] if r['family']==name);queue_path=Path(item['path']);queue=read(queue_path)
    validate_clone(read(ROOT/family['source_queue']),queue,family,proof['full_data_binding'])
    install_cached_loader(lambda snapshot:None)
    spec=dict(name=name,source_path=family['controller_source'],source_functions=['main'],queue_path=str(queue_path),queue_sha256=item['sha256'])
    controller,namespace,_=load_controller(ROOT,dict(source_receipts=config['source_receipts'],controllers=[spec]),name)
    base=ROOT/config['state_root']/'families'/name
    api=resource_api();owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full raw family owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous family attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),family=name,required_cases=family['required_cases'],
        state='starting_original_full_source_controller',active_pid=None,all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());update_file(base/'status.json',state)
    proxy=SimpleNamespace(**vars(subprocess))
    if name=='neural':
        from scripts.flow_matching import run_tsad_native_gpu_resource_queue_v1 as gpu_guard
        gpu_api=gpu_guard.guarded_api()
        registration=dict(settings=config['GPU_settings'],resource_lock=config['GPU_resource_lock'],model_worker='scripts.flow_matching.run_tsad_experiment')
        proxy.Popen=gpu_guard.guarded_factory(gpu_api,registration,update,queue_path,queue,base)
    else:
        guards=dict(vars(cpu_guard));node=capped_class_ast(Path(cpu_guard.__file__).read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(cpu_guard.__file__),'exec'),guards)
        factory=FunctionType(cpu_guard.guarded_factory.__code__,guards)
        proxy.Popen=factory(api,config['CPU_settings'],ROOT/config['CPU_resource_lock'],base,update,queue_path,queue)
    namespace['subprocess']=proxy
    sys.argv=[family['controller_source'],'--queue',str(queue_path)]
    try:
        update({});code=controller.main()
        completed=[];unresolved=[]
        for job in queue['jobs']:
            receipt=base/'receipts'/(job['id']+'.json')
            if namespace['complete_result'](ROOT,job) and receipt.exists():
                g=read(receipt)
                if g['exit_code']==0 and not g['resource_guard_aborted']:completed.append(job['id']);continue
            unresolved.append(job['id'])
        update(dict(state='original_source_controller_terminal',source_returncode=code,active_pid=None,
            original_training_and_scoring_complete=completed,unresolved_or_failed_preserved=unresolved,
            independent_model_replay_and_all_label_control_metrics_complete=False))
    finally:owner.close()


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full873materialization owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous full873owner attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=873,
        actual_model_jobs_started=False,all_original_experiments_complete=False)
    def update(**kw):state.update(kw,captured_utc=datetime.now(timezone.utc).isoformat());update_file(base/'status.json',state)
    try:
        while True:
            binding=ready_data(config)
            if binding is not None:break
            update(state='waiting_actual_full109_data_and_original_guard',active_pid=None);time.sleep(10)
        verify(config);proof=materialize(config,binding);verify_materialization(config)
        handles=[]
        for family in config['families']:
            path=base/'launches'/family['name'];path.mkdir(parents=True,exist_ok=False)
            command=[config['controller_python'],'-B','-X','utf8','-u','-m',MODULE,'--family',family['name']]
            with (path/'stdout.log').open('xb') as out,(path/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(family=family['name'],pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(path/'launch.json',handle);handles.append(handle)
            update(state='materialized_all873_and_launched_original_family_controllers',family_handles=handles,
                native_numerical_workers_unchanged=True,actual_model_jobs_started='check_individual_family_worker_handles')
        update(state='all7metadata_controllers_launched_actual_training_and_reviews_remain',family_handles=handles)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--family',choices=list(FAMILIES));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,full_original_cases=873)));return
    config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=873,per_dataset=291,families=7,
        actual_complete109data_ready=ready_data(config) is not None,torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
        actual_new_model_training_complete=False)))
    elif args.family:family_controller(config,args.family)
    else:controller(config)


if __name__=='__main__':main()
