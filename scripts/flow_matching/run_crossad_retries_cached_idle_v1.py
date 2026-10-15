"""Cold/cache three verified idle CrossAD recovery hosts, retaining all science."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as original

ROOT=original.ROOT
MODULE='scripts.flow_matching.run_crossad_retries_cached_idle_v1'
CONFIG='configs/runtime/fm_crossad_three_retries_cached_idle.v1.json'
TEST='tests/test_fm_crossad_three_retries_cached_idle_v1.py'
BASE='experiments/runs/fm_crossad_three_retries_cached_idle_v1'
SPECS=dict(
    gecco=('scripts.flow_matching.run_crossad_gecco1103_full_retry_v1','configs/runtime/fm_crossad_gecco1103_full20_retry.v1.json',33008,1791884424.1284602,
        'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'),
    psm=('scripts.flow_matching.crossad_psm1103_full_retry_v1','configs/runtime/fm_crossad_psm1103_full20_retry.v1.json',37628,1791907602.5502596,
        'C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'),
    smap=('scripts.flow_matching.crossad_smap1103_full_retry_v1','configs/runtime/fm_crossad_smap1103_full20_retry.v1.json',35944,1791909960.3811448,
        'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'))
ENTRIES={name:(value[0],139) for name,value in SPECS.items()}
read,sha=original.read,original.digest


def registration():
    roles={};pins={MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py','configs/models/fm_crossad_author_pipeline.json'}
    for name,(entry,path,pid,ctime,exe) in SPECS.items():
        production=read(ROOT/path);queue=read(ROOT/production['native_queue'])
        canonical=production['selected_original_ids'][0];job=next(j for j in queue['jobs'] if j['id']==canonical)
        roles[name]=dict(entry_module=entry,original_arguments=['--config',path],
            previous_owner=dict(pid=pid,create_time=ctime,command=[exe,'-X','utf8','-u','-m',entry,'--config',path]),
            previous_state_path=production['state_root']+'/status.json',original_runtime=path,
            original_runtime_sha256=sha(ROOT/path),original_queue=production['native_queue'],required_original_slots=139,
            canonical_id=canonical,output_directory=job['output_directory'],original_settings=production['settings'],
            original_shared_lock=production['shared_lock'])
        pins.update([path,production['native_queue'],entry.replace('.','/')+'.py'])
        pins.update(p['path'] for p in production['source_receipts'])
    return dict(schema_version=1,roles=roles,handoff_state_root=BASE,original_scope=dict(original.SCOPE),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        citations=read(ROOT/'configs/models/fm_crossad_author_pipeline.json')['citations'],all_original_experiments_complete=False,
        boundary='Same full139 CrossAD roster,threeexisting selected recoveries,all original20epoch/batch128/patience3 recipes. '
            'Exact entry modules/all arguments/model evaluators/40GiB admission/original8GiB emergency and locks retained. '
            'Only verified idle parents with absent new scientific outputs and no numeric descendants handed off. '
            'Cached equivalent Windows resource snapshot,unchanged guard functions. Existing failures/checkpoints preserved;not new training or seed slots. '
            'Full45/681/273/702research,TAB and native independent review obligations remain.')


def verify(config,name,root=ROOT):
    if config!=registration():raise ValueError('All three full139 bindings,original budgets/guards/commands and exact idle owners required')
    body=FunctionType(original.verify.__code__,dict(original.__dict__,ENTRIES=ENTRIES),argdefs=original.verify.__defaults__)
    return body(config,name,root)


def pending_unstarted(role):return not (ROOT/role['output_directory']).exists()


def safe_phase(role,state,expected,descendants):
    return (expected==role['previous_owner'] and state.get('create_time')==expected['create_time']
        and state.get('selected_original_id')==role['canonical_id']
        and state.get('configuration_sha256')==role['original_runtime_sha256']
        and state.get('original_max_epochs')==20 and state.get('credited_previous_epochs_for_fresh_retry')==0
        and original.idle_allowed(state,expected,descendants) and pending_unstarted(role))


def handoff(role,base,poll_seconds=10):
    ns=dict(original.__dict__,idle_allowed=lambda state,expected,desc:safe_phase(role,state,expected,desc))
    return FunctionType(original.handoff.__code__,ns,argdefs=original.handoff.__defaults__)(role,base,poll_seconds)


def namespace():
    return dict(original.__dict__,ENTRIES=ENTRIES,verify=verify,handoff=handoff)


def actual_guard_api_probe(config,name):
    role=verify(config,name)
    original.install_cached_loader(lambda snapshot:None)
    entry=importlib.import_module(role['entry_module']);production=read(ROOT/role['original_runtime'])
    queue,job,api=entry.verify(production)
    snapshot=api['resource_snapshot']
    if snapshot.__name__!='cached_snapshot':raise ValueError('Actual original CrossAD guard must resolve cached equivalent telemetry')
    import psutil
    return dict(role=name,original_jobs=len(queue['jobs']),selected=job['id'],settings=production['settings'],
        original_entry=role['entry_module'],original_arguments=role['original_arguments'],actual_guard_snapshot=snapshot.__name__,
        snapshot=snapshot(),private_bytes=psutil.Process().memory_info().private,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
        entry_main_executed=False,training_executed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=CONFIG)
    p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');p.add_argument('--role',choices=ENTRIES);a=p.parse_args()
    if a.register:
        c=registration();original.exclusive(ROOT/CONFIG,c);print(json.dumps(dict(registered=True,full_original_slots=139,selected_recovery_hosts=3)));return
    c=read(ROOT/a.config)
    if a.role is None:raise ValueError('Explicit registered role required')
    if a.probe_only:print(json.dumps(actual_guard_api_probe(c,a.role)));return
    argv=sys.argv
    try:
        sys.argv=[MODULE,'--config',a.config,'--role',a.role]
        FunctionType(original.main.__code__,namespace())()
    finally:sys.argv=argv


if __name__=='__main__':main()
