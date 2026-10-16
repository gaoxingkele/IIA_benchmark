"""Recover pre-worker hot-status sharing failures;keep all500models/215tasks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import run_anomaly_transformer_full_cohort_statistics_v1 as original
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as handoff
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

ROOT,read,sha=original.ROOT,original.read,original.sha
MODULE='scripts.flow_matching.recover_anomaly_transformer_full215_statistics_v2'
CONFIG='configs/runtime/fm_anomaly_transformer_full215_statistics_metadata_recovery.v2.json'
OLD_CONFIG='configs/runtime/fm_anomaly_transformer_full_cohort_statistics.v1.json'
SNAPSHOT='configs/reproducibility/fm_anomaly_transformer_statistics_preworker_failure_snapshot.2026-10-16_v1.json'
TEST='tests/test_fm_anomaly_transformer_full215_statistics_metadata_recovery_v2.py'


class HotStatusNotReady(Exception):
    """One exact producer telemetry file is currently being replaced or written."""


def only_preworker_metadata_failures(state,tasks,source):
    failures=state.get('failed_preserved',{})
    hot=(ROOT/source['state_root']/'status.json').resolve()
    allowed=set()
    for identity,error in failures.items():
        if (identity in tasks and tasks[identity]['kind']=='cohort' and isinstance(error,str)
                and error.startswith("[Errno 13] Permission denied: '") and error.endswith("'")):
            filename=error.split("Permission denied: '",1)[1][:-1]
            if Path(filename).resolve()==hot:allowed.add(identity)
    for identity,error in failures.items():
        if identity in allowed:continue
        if identity not in tasks or not isinstance(error,str):return False
        prefix='Preserved failed prerequisite statistics task: '
        if not error.startswith(prefix):return False
        dependency=error[len(prefix):]
        if dependency not in allowed or dependency not in tasks[identity].get('dependencies',[]):return False
    return bool(allowed)


def registration():
    c=deepcopy(read(ROOT/OLD_CONFIG));source,cases,_,tasks=original.verify(c)
    preserved=read(ROOT/SNAPSHOT)
    if (preserved.get('active_pid') is not None or preserved.get('completed_tasks')!=[]
            or preserved.get('state')!='waiting_all_five_actual_full_model_proofs'
            or not only_preworker_metadata_failures(preserved,tasks,source)):
        raise ValueError('Only proven pre-worker metadata/dependency failures with no completed task may recover')
    command=[c['python'],'-B','-X','utf8','-u','-m',original.MODULE,'--config',OLD_CONFIG]
    c.update(schema_version=2,predecessor_configuration=OLD_CONFIG,predecessor_configuration_sha256=sha(ROOT/OLD_CONFIG),
        preserved_preworker_failure_snapshot=SNAPSHOT,preserved_preworker_failure_snapshot_sha256=sha(ROOT/SNAPSHOT),
        predecessor_owner=dict(pid=preserved['pid'],create_time=preserved['create_time'],command=command),
        recovered_initial_preworker_task_ids=sorted(preserved['failed_preserved']),
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_full215_statistics_metadata_recovery_v2',
        state_root='experiments/runs/fm_anomaly_transformer_full215_statistics_metadata_recovery_v2',
        numerical_worker_is_frozen_v1_function=True,numerical_attempts_repeated=False)
    c['source_receipts']+=[dict(path=p,sha256=sha(ROOT/p)) for p in [OLD_CONFIG,SNAPSHOT,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py']]
    c['boundary']+=' Isolated metadata recovery preserves the9initial hot-status PermissionError/dependency failures. Replace only a verified idle original statistics owner with no numerical attempt,artifact or child. Same original500models/all215tasks,worker code,formula/calibration/uncertainty and32GiBphysical/40GiBcommit/16GiBemergency policy. Only PermissionError,partial JSON or replacement disappearance on the one live producer status file becomes not-ready;immutable proof/guard/data failures remain fatal. No model/statistics numerical attempt is retried.'
    return c


def verify(c):
    if c!=registration():raise ValueError('All original500models/215statistics and exact metadata-only recovery contract required')
    return original.verify(c)


def hot_status_read(path,hot,reader):
    if Path(path).resolve()!=Path(hot).resolve():return reader(path)
    try:return reader(path)
    except (PermissionError,FileNotFoundError,json.JSONDecodeError) as error:
        raise HotStatusNotReady(str(error)) from error


def readiness(runtime,source,task,done,failed):
    hot=ROOT/source['state_root']/'status.json'
    namespace=dict(original.readiness.__globals__,read=lambda p:hot_status_read(p,hot,original.read))
    unchanged=FunctionType(original.readiness.__code__,namespace,original.readiness.__name__)
    try:return unchanged(runtime,source,task,done,failed)
    except HotStatusNotReady:return False


def no_original_attempts(c):
    old=read(ROOT/OLD_CONFIG)
    return not any((ROOT/old['state_root']/'attempts'/t['id']).exists()
        or (Path(old['artifact_root'])/t['id']).exists() for t in old['tasks'])


def idle_allowed(state,expected,descendants):
    c=read(ROOT/OLD_CONFIG);source=read(ROOT/original.PRODUCER);tasks={t['id']:t for t in c['tasks']}
    return (state.get('pid')==expected['pid'] and state.get('create_time')==expected['create_time']
        and state.get('active_pid') is None and state.get('completed_tasks')==[]
        and state.get('state')=='waiting_all_five_actual_full_model_proofs'
        and only_preworker_metadata_failures(state,tasks,source) and no_original_attempts(c)
        and all(handoff.allowed_console_child(p['name'],p['exe']) for p in descendants))


def controller(runtime,source,selected):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/runtime['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'recovery_handoff.lock')
    if owner is None:raise RuntimeError('A complete215metadata successor already owns this handoff')
    oldlock=None
    try:
        if (base/'status.json').exists():raise FileExistsError('Preserve any previous metadata/statistics attempt')
        role=dict(previous_owner=runtime['predecessor_owner'],previous_state_path=read(ROOT/OLD_CONFIG)['state_root']+'/status.json')
        transfer=FunctionType(handoff.handoff.__code__,dict(handoff.__dict__,idle_allowed=idle_allowed))
        transfer(role,base)
        if not no_original_attempts(runtime):raise RuntimeError('Original numerical attempt appeared;never duplicate it')
        oldlock=api['exclusive_lock'](ROOT/read(ROOT/OLD_CONFIG)['state_root']/'controller.lock')
        if oldlock is None:raise RuntimeError('Another predecessor owner appeared;preserve without duplicating tasks')
        unchanged=FunctionType(original.controller.__code__,dict(original.controller.__globals__,MODULE=MODULE,readiness=readiness))
        unchanged(runtime,source,selected)
    finally:
        if oldlock is not None:oldlock.close()
        owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--config',default=CONFIG)
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--task-id')
    a=parser.parse_args()
    if a.config!=CONFIG:raise ValueError('Use only the isolated complete215 recovery configuration')
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/a.config);source,cases,contract,selected=verify(c);c['_config_path']=a.config
    if a.probe_only:
        import sys
        print(json.dumps(dict(passed=True,required_models=500,tasks=len(selected),
            task_kinds={k:sum(t['kind']==k for t in selected.values()) for k in ['cohort','paired','macro']},
            preserved_preworker_failures=len(c['recovered_initial_preworker_task_ids']),
            original_worker_bytecode_unchanged=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            numerical_worker_started=False)))
    elif a.worker:original.worker(c,source,cases,contract,selected[a.task_id],selected)
    else:controller(c,source,selected)


if __name__=='__main__':main()
