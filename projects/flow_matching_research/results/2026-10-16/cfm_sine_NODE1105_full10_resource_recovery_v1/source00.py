"""One explicit full-budget recovery of the terminal appendix sine NODE abort.

CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ. Frozen local paper-derived
implementation only; author equivalence remains unproved. No automatic retry.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as cold
from scripts.flow_matching.run_light_controller import load_functions

ROOT,read,sha,write=cold.ROOT,cold.read,cold.sha,cold.write
MODULE='scripts.flow_matching.run_cfm_sine_NODE1105_resource_recovery_v1'
CONFIG='configs/runtime/fm_cfm_sine_NODE1105_resource_recovery.v1.json'
QUEUE='configs/experiments/fm_cfm_sine_NODE1105_resource_recovery_queue.v1.json'
REPLAY='configs/runtime/fm_cfm_sine_NODE1105_resource_recovery_full285_replay.v1.json'
ORIGINAL='configs/experiments/fm_cfm_ts_original_queue.v1.json'
PRIMARY='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
CANONICAL='cfm_ts_original__appendix_d__node_e10__sine__seed1105'
TEST='tests/test_fm_cfm_sine_NODE1105_resource_recovery_v1.py'
FAILED_HANDLE=dict(pid=12932,create_time=1792074904.6385772)
BASE='experiments/runs/fm_cfm_sine_NODE1105_resource_recovery_v1'
ARTIFACT='F:/aicoding/IIA_Data/experiments/cfm_sine_NODE1105_resource_recovery_v1'
EXECUTION_FIELDS={'id','output_directory','artifact_directory','recovery_original_job_id'}
load_functions(ROOT,'scripts/flow_matching/run_cfm_ts_original_job.py',['fingerprint'],globals())


def science(job):return {k:v for k,v in job.items() if k not in EXECUTION_FIELDS}


def exclusive(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def failed_evidence(source):
    import psutil
    job=next(j for j in source['jobs'] if j['id']==CANONICAL)
    receipt=source['state_root']+'/receipts/'+CANONICAL+'.json'
    progress=job['output_directory']+'/progress.json'
    guard,p=read(ROOT/receipt),read(ROOT/progress)
    if (guard['complete'] or guard['exit_code']==0 or not guard['resource_observations']['resource_guard_aborted']
            or p['required_epochs']!=10 or p['epoch']!=1 or p['optimizer_updates']!=2):
        raise ValueError('Exact terminal resource-aborted1/10 epochs and2/20 updates required')
    try:
        actor=psutil.Process(FAILED_HANDLE['pid'])
        if abs(actor.create_time()-FAILED_HANDLE['create_time'])<.01:raise ValueError('Original scientific worker remains alive')
    except psutil.NoSuchProcess:pass
    if (ROOT/job['output_directory']/'result.json').exists():raise ValueError('Never repeat completed original science')
    for directory in [ROOT/job['output_directory'],Path(job['artifact_directory'])]:
        if directory.exists() and any(p.is_file() and p.suffix in ['.pt','.pth','.ckpt','.pkl'] for p in directory.rglob('*')):
            raise ValueError('A saved native training state requires its own restoration audit')
    return job,receipt,progress


def registration():
    source=read(ROOT/ORIGINAL);cold.verify(source);canonical,receipt,progress=failed_evidence(source)
    queue=deepcopy(source);job=next(j for j in queue['jobs'] if j['id']==CANONICAL)
    job.update(id=CANONICAL+'__resource_checkpointed_v1',recovery_original_job_id=CANONICAL,
        output_directory=BASE+'/job',artifact_directory=ARTIFACT+'/source')
    policy=dict(source['resource_policy'],minimum_commit_headroom_bytes=32*2**30)
    sources={ORIGINAL,receipt,progress,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py',
        'scripts/flow_matching/cfm_complete_epoch_checkpoint_v1.py',
        'scripts/flow_matching/run_cfm_metadata_permission_replay_queue_v1.py',
        'scripts/flow_matching/run_cfm_uncovered_recovery_completion_v1.py',
        'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',PRIMARY}
    pins=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)]
    queue.update(state_root=BASE,resource_policy=policy,recovery_original_queue=ORIGINAL,
        recovery_original_queue_sha256=sha(ROOT/ORIGINAL),recovery_job_ids=[job['id']],
        source_receipts=source['source_receipts']+pins)
    config=dict(queue=QUEUE,job_id=job['id'],canonical_job=canonical,source_receipts=pins,
        original_queue=ORIGINAL,state_root=BASE,artifact_root=ARTIFACT,
        resource_policy=policy,family_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',
        stable_admission_samples=6,stable_poll_seconds=5,checkpoint_interval=1,
        CPU_only_environment=dict(CUDA_VISIBLE_DEVICES='-1'),
        maximum_new_attempts=1,partial_epochs_credited=0,failed_handle=FAILED_HANDLE,
        preserved_failed_receipt=receipt,preserved_failed_progress=progress,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
        reproduction_status='executable_paper_derived_reconstruction_not_author_equivalent',
        all_original_experiments_complete=False)
    return config,queue


def verify(config):
    expected,queue=registration()
    if config!=expected or read(ROOT/QUEUE)!=queue:raise ValueError('Exact full10epochs/20updates/80x500tests and frozen recovery contract required')
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if science(job)!=science(config['canonical_job']):raise ValueError('Every scientific field must remain unchanged')
    return [job],queue


def worker_command(config):
    queue=read(ROOT/config['queue'])
    return [str(ROOT/queue['python']),'-B','-X','utf8','-u','-m',MODULE,'--stage','train']


def training_worker(config):
    jobs,queue=verify(config);job=jobs[0]
    os.environ.update(config['CPU_only_environment'])
    from scripts.flow_matching import run_cfm_checkpointed_original_job_v1 as source
    from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import CompleteEpochLane,bind_fit
    def bind(model,actual,artifacts):return bind_fit(model,CompleteEpochLane(actual,artifacts,interval=1))
    values=dict(source.original.__dict__,_bind_complete_epoch_fit=bind)
    exec(compile(source.worker_ast(inspect.getsource(source.original.run)),source.original.__file__,'exec'),values)
    values['run'](queue,job)
    ledger=read(Path(job['artifact_directory'])/'completed_full_epoch_checkpoints.json')
    if [r['completed_original_epochs'] for r in ledger]!=list(range(1,11)):
        raise ValueError('All10 complete original epoch states required')
    for index,r in enumerate(ledger,1):
        if (r['optimizer_updates']!=index*2 or not r['all_native_model_optimizer_rng_gradients_history_and_order_values_reread_bitwise']
                or not r['no_training_state_or_random_state_changed_by_save'] or r['additional_optimizer_updates']!=0
                or sha(r['full_state_path'])!=r['full_state_sha256']):raise ValueError('Every full native epoch state must remain exact')
    if not source.original.complete(job):raise ValueError('Full original source completion required')


def source_closed(config):
    path=ROOT/config['state_root']/'receipt.json'
    if not path.exists():return False
    guard=read(path);jobs,queue=verify(config);job=jobs[0]
    if (not guard['complete'] or guard['exit_code']!=0 or guard['resource_observations']['resource_guard_aborted']
            or guard['exact_child_handle']['command']!=worker_command(config)):return False
    resultpath=ROOT/job['output_directory']/'result.json'
    if sha(resultpath)!=guard['result_sha256']:raise ValueError('Completed source result changed')
    result=read(resultpath)
    if (result['job']!=job or result['job_sha256']!=fingerprint(job) or result['epochs_completed']!=10
            or result['optimizer_updates']!=20 or result['test_prediction_shape']!=[80,500,1]):
        raise ValueError('Complete frozen source,all20updates and40000testcoordinates required')
    return True


def stable_headroom(config,api,update):
    count=0;policy=config['resource_policy']
    while count<config['stable_admission_samples']:
        snap=api['resource_snapshot']()
        enough=snap['physical_available_bytes']>=policy['minimum_available_memory_bytes'] and snap['commit_available_bytes']>=policy['minimum_commit_headroom_bytes']
        count=count+1 if enough else 0
        update(dict(state='waiting_stable_original_family_resources',stable_samples=count,**snap))
        if count<config['stable_admission_samples']:time.sleep(config['stable_poll_seconds'])


def controller(config):
    import psutil
    verify(config);api=cold.resource_api()
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api['resource_snapshot']=resource_snapshot
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing explicit recovery owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous recovery attempt')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None,
        canonical_job_id=CANONICAL,all_original_experiments_complete=False,partial_epochs_credited=0)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        jobs,queue=verify(config);job=jobs[0]
        if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():raise FileExistsError('Preserve all previous scientific outputs')
        with api['resource_slot'](ROOT/config['family_lock'],config['resource_policy'],update):
            stable_headroom(config,api,update);verify(config)
            command=worker_command(config)
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',**config['CPU_only_environment']),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                update(dict(state='running_full_original10_checkpointed_epochs',active_pid=child.pid,exact_child_handle=handle))
                code,usage=api['guarded_wait'](child,config['resource_policy'],update)
        resultpath=ROOT/job['output_directory']/'result.json';result=read(resultpath) if resultpath.exists() else None
        passed=bool(code==0 and not usage['resource_guard_aborted'] and result and result['status']=='completed'
            and result['epochs_completed']==10 and result['optimizer_updates']==20 and result['job_sha256']==fingerprint(job))
        exclusive(config['state_root']+'/receipt.json',dict(exit_code=code,complete=passed,resource_observations=usage,
            exact_child_handle=handle,result_sha256=sha(resultpath) if resultpath.exists() else None))
        update(dict(state='completed_source_native_and_independent_reviews_pending' if passed else 'failed_attempt_preserved',active_pid=None))
    finally:owner.close()


def replay_namespace():
    from scripts.flow_matching import run_cfm_metadata_permission_replay_queue_v1 as previous
    ns=dict(previous.__dict__,PRODUCER=CONFIG,producer=SimpleNamespace(verify_runtime=verify))
    values=FunctionType(previous.namespace.__code__,ns)()
    selected=values['_selected_ready']
    values['_selected_ready']=lambda route,runtime:selected(route,runtime) and source_closed(read(ROOT/CONFIG))
    return values


def replay_registration(config,queue):
    from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as supplement
    primary=read(ROOT/PRIMARY);value=deepcopy(primary)
    value.update(previous_full285_configuration=PRIMARY,producer_configuration=CONFIG,
        runtime_configuration=REPLAY,routes=supplement.expected_routes(primary,queue),selected_canonical_ids=[CANONICAL],
        state_root=BASE+'/replay',child_state_root=BASE+'/replay/children',
        child_config_root='configs/reproducibility/fm_cfm_sine_NODE1105_recovery_v1',artifact_root=ARTIFACT+'/replay',
        source_receipts=primary['source_receipts']+config['source_receipts']+
            [dict(path=p,sha256=sha(ROOT/p)) for p in [CONFIG,QUEUE]],
        boundary='All285 canonical slots retained;only the terminal resource-aborted appendix sine NODE1105 route changes. '
            'Other primary certificates adopted without duplicate execution. Exact source guard required;original native/independent workers,all80x500coordinates,tolerance and resources unchanged.')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--stage',choices=['train','replay']);a=p.parse_args()
    if a.register:
        c,q=registration();exclusive(QUEUE,q);exclusive(CONFIG,c);verify(c)
        r=replay_registration(c,q);replay_namespace()['verify'](r);exclusive(REPLAY,r)
        print(json.dumps(dict(canonical_jobs=285,selected_recovery=1,epochs=10,updates=20,test_points=40000)));return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:
        replay_namespace()['verify'](read(ROOT/REPLAY))
        print(json.dumps(dict(passed=True,source_closed=source_closed(c),numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif a.stage=='train':training_worker(c)
    elif a.stage=='replay':replay_namespace()['controller'](read(ROOT/REPLAY),REPLAY)
    else:controller(c)


if __name__=='__main__':main()
