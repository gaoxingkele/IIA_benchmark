"""Execute all5 full-SMAP DCD105 controls on the audited common105 split.

Preserve old attempts; this is a distinct preprocessing control. Matching other
methods to this protocol remains required before any shared leaderboard claim.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import prepare_original_full_entity_tsad_min105_v3 as prep
from scripts.flow_matching import run_original_full_raw873_experiments_v2 as original

ROOT,read,sha,write=prep.ROOT,prep.read,prep.sha,prep.write
MODULE='scripts.flow_matching.run_full_smap_DCD105_min105_controls_v1'
CONFIG='configs/reproducibility/fm_full_smap_DCD105_min105_controls.2026-10-16_v1.json'
TEST='tests/test_fm_full_smap_DCD105_min105_controls_v1.py'


def source():
    config=read(ROOT/original.CONFIG)
    path=Path(config['artifact_root'])/'queues/baselines.json'
    queue=read(path);jobs=[]
    for job in queue['jobs']:
        model=read(ROOT/job['model_config'])
        if job['dataset']=='smap' and model['parameters']['method']=='dcdetector':jobs.append(job)
    if len(jobs)!=5 or {j['seed'] for j in jobs}!=set(range(1103,1108)) or any(j['window']!=105 for j in jobs):
        raise ValueError('All5original SMAP DCD105 seeds required')
    return path,queue,jobs


def registration():
    path,queue,jobs=source();policy=read(ROOT/original.CPU_POLICY)
    pins={str(path),original.CONFIG,prep.CONFIG,prep.MODULE.replace('.','/')+'.py',
        MODULE.replace('.','/')+'.py',TEST,original.CPU_POLICY,
        'scripts/flow_matching/run_original_full_raw873_experiments_v2.py',
        'scripts/flow_matching/run_tsad_queue.py','scripts/flow_matching/run_tsad_experiment.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'scripts/flow_matching/run_bounded_strict_cpu_queue_v2.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py'}
    from scripts.flow_matching.run_light_controller import FUNCTIONS
    pins.update(FUNCTIONS);pins.update(j['model_config'] for j in jobs)
    return dict(schema_version=1,full_entity_preparation=prep.CONFIG,source_queue=str(path),
        selected_source_ids=[j['id'] for j in jobs],required_cases=5,dataset='smap',entities=54,
        window=105,seeds=list(range(1103,1108)),original_evaluation=queue['evaluation'],
        python=queue['python'],controller_python=sys.executable,
        settings=policy['settings'],resource_lock=policy['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/full_smap_DCD105_min105_controls_v1',
        state_root='experiments/runs/fm_full_smap_DCD105_min105_controls_v1',
        boundary='All5source seeds/full54SMAP entities/original DCD105 model and complete epochs,batch,width,loss. '
            'Common105validation data audit required before original numerical worker;no source retry or mutation. '
            'Original strict16/32/16GiB CPU,10/12GiBworker caps and shared lock retained. '
            'Control results cannot be pooled with V2scores as a shared protocol;matched other-method controls, '
            'independent native/label reviews and paired5seed statistics remain separate requirements. '
            'All873/2037and original45papers/681records/273data tasks/702targets remain unfinished broader scope.')


def verify(c):
    if c!=registration():raise ValueError('All5fullSMAP original DCD105 jobs and resource policy required')


def binding(c):
    result=FunctionType(original.ready_data.__code__,dict(original.ready_data.__globals__,preparation=prep))(c)
    if result is None:return None
    report=read(Path(read(ROOT/prep.CONFIG)['artifact_root'])/prep.REPORT)
    cross=report['cross_version']
    if (not cross['passed'] or cross['entities']!=109 or cross['unchanged_entities']!=107
            or set(cross['changed_entities'])!=set(prep.CHANGED) or cross['all_test_label_overlays_verified']!=298
            or report['old_scores_same_protocol_certified']):
        raise ValueError('Actual complete common105 preparation/cross-version proof required')
    return result


def clone_queue(c,b):
    _,old,jobs=source();queue=deepcopy(old);new=[]
    for job in jobs:
        changed=deepcopy(job);identity=job['id']+'__min105_control_v1';record=b['records']['smap']
        changed.update(id=identity,dataset_manifest_path=b['manifest'],
            output_directory=c['artifact_root']+'/jobs/'+identity,
            common105_original_source_id=job['id'],common105_preprocessing_control=True)
        pins={p['path']:p['sha256'] for p in job['frozen_files']}
        pins.update({b['manifest']:b['manifest_sha256'],record['input_npz']:record['input_sha256'],CONFIG:sha(ROOT/CONFIG)})
        changed['frozen_files']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())];new.append(changed)
    queue.update(jobs=new,state_root=c['artifact_root']+'/source_controller',
        boundary=old['boundary']+'; New common105 preprocessing control;not interchangeable with original V2 predictions.')
    validate_clone(c,queue,b)
    return queue


def validate_clone(c,queue,b):
    _,old,jobs=source();allowed={'id','dataset_manifest_path','output_directory','frozen_files',
        'common105_original_source_id','common105_preprocessing_control'}
    if len(queue['jobs'])!=5:raise ValueError('Every original seed required')
    for a,z in zip(jobs,queue['jobs']):
        if (z['id']!=a['id']+'__min105_control_v1' or z['common105_original_source_id']!=a['id']
                or z['dataset_manifest_path']!=b['manifest'] or not z['common105_preprocessing_control']
                or z['output_directory']!=c['artifact_root']+'/jobs/'+z['id']
                or {k:v for k,v in a.items() if k not in allowed}!={k:v for k,v in z.items() if k not in allowed}):
            raise ValueError('Only explicit new data/output/provenance fields may change')
        pins={r['path']:r['sha256'] for r in z['frozen_files']};record=b['records']['smap']
        needed={p['path']:p['sha256'] for p in a['frozen_files']}
        needed.update({b['manifest']:b['manifest_sha256'],record['input_npz']:record['input_sha256'],CONFIG:sha(ROOT/CONFIG)})
        if pins!=needed:raise ValueError('All old pins and complete new data must remain frozen')
    if any(queue.get(k)!=v for k,v in old.items() if k not in ['jobs','state_root','boundary']):
        raise ValueError('Original evaluation/environment/source phase gates must remain')


def controller(c):
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    from scripts.flow_matching.run_light_controller import load_controller
    from scripts.flow_matching import run_resource_gated_light_controller_v1 as guard
    from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import capped_class_ast
    api=resource_api();base=ROOT/c['state_root'];owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing common105 owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous original source attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=5,active_pid=None,all_original_experiments_complete=False)
    def update(v):
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());original.update_file(base/'status.json',state)
    try:
        verify(c);b=binding(c)
        if b is None:raise ValueError('Actual complete105 data required before launching source')
        queue=clone_queue(c,b);path=Path(c['artifact_root'])/'queue.json';write(path,queue)
        write(path.parent/'materialization.json',dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),
            queue=str(path),queue_sha256=sha(path),full_data_binding=b,all_original_experiments_complete=False))
        spec=dict(name='full54_SMAP_DCD105_common105',source_path='scripts/flow_matching/run_tsad_queue.py',
            source_functions=['main'],queue_path=str(path),queue_sha256=sha(path))
        source_controller,namespace,_=load_controller(ROOT,dict(source_receipts=c['source_receipts'],controllers=[spec]),spec['name'])
        install_cached_loader(lambda v:None)
        guards=dict(vars(guard));node=capped_class_ast(Path(guard.__file__).read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(guard.__file__),'exec'),guards)
        factory=FunctionType(guard.guarded_factory.__code__,guards)
        proxy=SimpleNamespace(**vars(subprocess));proxy.Popen=factory(api,c['settings'],ROOT/c['resource_lock'],base,update,path,queue)
        namespace['subprocess']=proxy;sys.argv=['original-full54-DCD105-common105','--queue',str(path)]
        update(dict(state='starting_unchanged_original_DCD105_source'))
        code=source_controller.main();completed=[];failed=[]
        for job in queue['jobs']:
            receipt=base/'receipts'/(job['id']+'.json')
            if namespace['complete_result'](ROOT,job) and receipt.exists():
                g=read(receipt)
                if g['exit_code']==0 and not g['resource_guard_aborted']:completed.append(job['id']);continue
            failed.append(job['id'])
        update(dict(state='source_terminal',source_returncode=code,active_pid=None,
            actual_source_cases_complete=completed,failed_preserved=failed,
            independent_native_and_all_label_reviews_complete=False))
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=5)));return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=5,entities=54,actual_full109_min105_data_ready=binding(c) is not None,
        torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules)))
    else:controller(c)


if __name__=='__main__':main()
