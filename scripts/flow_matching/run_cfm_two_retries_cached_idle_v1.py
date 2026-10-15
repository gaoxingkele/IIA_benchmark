"""Cache telemetry for two existing idle full-budget CFM recovery hosts."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as original

ROOT,read,sha=original.ROOT,original.read,original.digest
MODULE='scripts.flow_matching.run_cfm_two_retries_cached_idle_v1'
CONFIG='configs/runtime/fm_cfm_two_retries_cached_idle.v1.json'
TEST='tests/test_fm_cfm_two_retries_cached_idle_v1.py'
BASE='experiments/runs/fm_cfm_two_retries_cached_idle_v1'
ENTRY='scripts.flow_matching.run_cfm_ts_stricter_admission_retry_v1'
SPECS=dict(
    node400_1104=('configs/runtime/fm_cfm_seed1104_actual99_full_retry.v1.json',40980,1791885460.1111372,
        'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'),
    node600_1107=('configs/runtime/fm_cfm_node600_seed1107_actual41_full_retry.v1.json',24872,1791907602.53649,
        'C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'))
ENTRIES={name:(ENTRY,285) for name in SPECS}


def registration():
    roles={};pins={MODULE.replace('.','/')+'.py',TEST,ENTRY.replace('.','/')+'.py',
        'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py','scripts/flow_matching/cached_resource_snapshot_v1.py'}
    for name,(path,pid,ctime,exe) in SPECS.items():
        runtime=read(ROOT/path);selected=runtime['controllers']['recovery'];queue=read(ROOT/selected['queue'])
        if len(selected['selected_job_ids'])!=1:raise ValueError('Exactly one previously selected recovery per lane required')
        job=next(j for j in queue['jobs'] if j['id']==selected['selected_job_ids'][0])
        args=['--runtime-config',path,'--controller','recovery']
        roles[name]=dict(entry_module=ENTRY,original_arguments=args,
            previous_owner=dict(pid=pid,create_time=ctime,command=[exe,'-X','utf8','-u','-m',ENTRY,*args]),
            previous_state_path=queue['state_root']+'/status.json',original_runtime=path,
            original_runtime_sha256=sha(ROOT/path),original_queue=selected['queue'],queue_sha256=selected['queue_sha256'],
            required_original_slots=285,selected_job=job,resource_policy=queue['resource_policy'],
            stronger_admission_commit_bytes=runtime['retry_admission_commit_headroom_bytes'],family_lock=runtime['family_lock'])
        pins.update([path,selected['queue']]);pins.update(p['path'] for p in runtime['source_receipts'])
    return dict(schema_version=1,roles=roles,handoff_state_root=BASE,original_scope=dict(original.SCOPE),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',all_original_experiments_complete=False,
        boundary='Every285 canonical CFM slot retained;two already registered exact-budget NODE400seed1104/NODE600seed1107 recoveries only. '
            'Same entry/allargv/science/CPU1/12GiBphysical/40GiBadmission/16GiBemergency/familylock. '
            'Original unsuccessful99/41epoch incidents preserved and receive no fresh training credit. '
            'Equivalent cached resource snapshot only;no numerical child or source edited. '
            'Only exact idle legacy PID/state/queue/runtime identities with absent selected outputs/no numeric descendants handed off. '
            'All45/681/273/702research and source fidelity limits remain.')


def verify(c,name,root=ROOT):
    if c!=registration():raise ValueError('Both complete285 lanes,exact original commands,science and guards required')
    body=FunctionType(original.verify.__code__,dict(original.__dict__,ENTRIES=ENTRIES),argdefs=original.verify.__defaults__)
    return body(c,name,root)


def pending_unstarted(role):
    queue=read(ROOT/role['original_queue']);job=role['selected_job']
    receipt=ROOT/queue['state_root']/'receipts'/(job['id']+'.json')
    return not receipt.exists() and not (ROOT/job['output_directory']).exists() and not Path(job['artifact_directory']).exists()


def safe_phase(role,state,expected,desc):
    return (expected==role['previous_owner'] and state.get('queue_sha256')==role['queue_sha256']
        and state.get('runtime_sha256')==role['original_runtime_sha256'] and state.get('jobs')=={}
        and state.get('active_job') is None and original.idle_allowed(state,expected,desc) and pending_unstarted(role))


def handoff(role,base,poll_seconds=10):
    ns=dict(original.__dict__,idle_allowed=lambda state,expected,desc:safe_phase(role,state,expected,desc))
    return FunctionType(original.handoff.__code__,ns,argdefs=original.handoff.__defaults__)(role,base,poll_seconds)


def namespace():return dict(original.__dict__,ENTRIES=ENTRIES,verify=verify,handoff=handoff)


def actual_guard_api_probe(c,name):
    role=verify(c,name);original.install_cached_loader(lambda snapshot:None)
    entry=importlib.import_module(ENTRY);runtime=read(ROOT/role['original_runtime'])
    selected,queue=entry.original.binding(runtime,'recovery');api=entry.original.resource_api()
    snapshot=api['resource_snapshot']
    if snapshot.__name__!='cached_snapshot':raise ValueError('Actual original CFM guard must use cached telemetry')
    import psutil
    return dict(role=name,original_jobs=len(queue['jobs']),selected=selected['selected_job_ids'],
        original_policy=queue['resource_policy'],stronger_commit=runtime['retry_admission_commit_headroom_bytes'],
        actual_guard_snapshot=snapshot.__name__,snapshot=snapshot(),private_bytes=psutil.Process().memory_info().private,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,numerical_execution=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=CONFIG);p.add_argument('--role',choices=ENTRIES)
    p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:original.exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,original_slots=285,existing_recovery_lanes=2)));return
    c=read(ROOT/a.config)
    if a.role is None:raise ValueError('Explicit original recovery role required')
    if a.probe_only:print(json.dumps(actual_guard_api_probe(c,a.role)));return
    argv=sys.argv
    try:
        sys.argv=[MODULE,'--config',a.config,'--role',a.role]
        FunctionType(original.main.__code__,namespace())()
    finally:sys.argv=argv


if __name__=='__main__':main()
