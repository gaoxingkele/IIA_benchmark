"""Continue every original TAB878 fragment, yielding the audit lock between entities.

Original metric workers/configuration/commands and all numerical definitions
remain unchanged. Handoff waits for a genuinely terminal entity, never kills it.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as original
from scripts.flow_matching.run_idle_exact_command_cached_handoff_v3 import allowed_console_child

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.continue_TAB878_fair_entities_v1'
CONFIG='configs/runtime/fm_TAB878_fair_entity_continuation.v1.json'
TEST='tests/test_fm_TAB878_fair_entity_continuation_v1.py'
BASE='experiments/runs/fm_TAB878_fair_entity_continuation_v1'
EXPECTED=dict(pid=6120,create_time=1792096884.1711507,command=[
    'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe','-B','-X','utf8','-u','-m',original.MODULE])


def exclusive(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def registration():
    c=read(ROOT/original.CONFIG)
    pins={original.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        original.MODULE.replace('.','/')+'.py','scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py'}
    return dict(schema_version=1,original_configuration=original.CONFIG,original_configuration_sha256=sha(ROOT/original.CONFIG),
        state_root=BASE,previous_owner=deepcopy(EXPECTED),required_cases=878,required_label_views=2057,
        settings=c['settings'],resource_lock=c['resource_lock'],cpu_threads=c['CPU_threads'],
        handoff_poll_seconds=1,inter_entity_yield_seconds=20,maximum_new_attempts_per_fragment=1,
        retain_all_old_successful_and_failed_outcomes=True,scientific_artifact_root=c['artifact_root'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        citations=c['citations'],original_scope=c['original_scope'],all_original_experiments_complete=False,
        boundary='Original878/2057/all21functions/201calls/3VAL+10diagnostics/fullVUS/200entityCI unchanged. '
            'Original worker/config/command/array hashes and16/28/16GiB3/6caps/auditlock retained. '
            'Wait for completed actual child+guard,acquire actual audit lock before twice checking idle metadata owner;never stop numeric child. '
            'Resume and revalidate every actual guarded fragment,negative outcomes preserved,no retry. '
            'Release audit lock then yield20seconds between newly completed fragments to allow other original waiters to poll. '
            'Mirror current metadata status for existing resource-gap recovery observer;preserve old snapshot. '
            'No complete research/paper-equivalence claim;all45/681/273/702requirements remain.')


def verify(config):
    if config!=registration():raise ValueError('All878/2057/frozen metric program/resources and full inter-entity yield required')
    c=read(ROOT/original.CONFIG);original.verify(c);return c


def terminal_child(c,state):
    """A stale active_pid is accepted only with its exact terminal guard."""
    import psutil
    if state.get('active_pid') is None:return True
    entries=[e for e in c['entries'] if e['case']['id']==state.get('active_case')]
    if len(entries)!=1 or not state.get('active_entity_prefix'):return False
    entry=entries[0];prefix=state['active_entity_prefix']
    attempt,_=original.paths(c,entry,prefix);path=attempt/'guard_receipt.json'
    if not path.exists():return False
    g=read(path);h=g['exact_child_handle']
    if h['pid']!=state['active_pid'] or h['command']!=original.worker_command(c,entry,prefix):return False
    try:
        p=psutil.Process(h['pid'])
        if p.create_time()==h['create_time']:return False
    except psutil.NoSuchProcess:pass
    if not isinstance(g.get('exit_code'),int):return False
    if g['passed'] and prefix!='summary':original.checked_fragment(c,entry,prefix)
    return True


def boundary_allowed(state,expected,descendants,child_terminal):
    return (state.get('pid')==expected['pid'] and state.get('create_time')==expected['create_time']
        and state.get('required_cases')==878 and state.get('required_label_views')==2057 and child_terminal
        and all(allowed_console_child(d['name'],d['exe']) for d in descendants)
        and state.get('state') in {'between_original_entities','between_complete_TAB_cases','waiting_for_heavy_lock',
            'waiting_for_memory','waiting_actual_native_and_fresh_all_label_reviews','original_TAB_failure_preserved',
            'replaying_full_checkpoints','running_original_TAB_metric_suite'})


def handoff(config,c,api,base,update):
    import psutil
    expected=config['previous_owner'];statuspath=ROOT/c['state_root']/'status.json'
    while True:
        p=psutil.Process(expected['pid'])
        if p.create_time()!=expected['create_time'] or p.cmdline()!=expected['command']:
            raise ValueError('Previous exact TAB metadata owner changed;preserve all work')
        # If a metric is active its original parent owns this lock. We cannot
        # acquire it, and we do not suspend or modify that numerical process.
        slot=api['exclusive_lock'](ROOT/c['resource_lock'])
        if slot is None:
            update(state='waiting_actual_metric_lock_release_no_process_stopped');time.sleep(config['handoff_poll_seconds']);continue
        stopped=False;paused=False;oldlock=None
        try:
            state=read(statuspath)
            desc=[dict(pid=v.pid,name=v.name(),exe=v.exe()) for v in p.children(recursive=True)]
            if not boundary_allowed(state,expected,desc,terminal_child(c,state)):continue
            p.suspend();paused=True
            if p.create_time()!=expected['create_time'] or p.cmdline()!=expected['command']:raise ValueError('Suspended metadata identity changed')
            state=read(statuspath);desc=[dict(pid=v.pid,name=v.name(),exe=v.exe()) for v in p.children(recursive=True)]
            if not boundary_allowed(state,expected,desc,terminal_child(c,state)):continue
            exclusive(base/'previous_status_at_handoff.json',state)
            with (base/'previous_status_original_bytes.json').open('xb') as f:f.write(statuspath.read_bytes())
            exclusive(base/'handoff_identity.json',dict(previous_owner=expected,descendants=desc,actual_audit_lock_owned=True,
                exact_os_identity_checked_twice=True,actual_last_child_terminal=True,numerical_child_stopped=False))
            p.terminate();p.wait(timeout=15);stopped=True
            oldlock=api['exclusive_lock'](ROOT/c['state_root']/'controller.lock')
            if oldlock is None:raise RuntimeError('Another original TAB owner appeared;do not duplicate work')
            exclusive(base/'handoff.json',dict(previous_owner=expected,successor_pid=os.getpid(),waiting_metadata_only_stopped=True,
                original_numeric_child_stopped=False,captured_utc=datetime.now(timezone.utc).isoformat()))
            return state,oldlock
        finally:
            if paused and not stopped and p.is_running():p.resume()
            slot.close()
            if not stopped:time.sleep(config['handoff_poll_seconds'])


def fragment_action(c,entry,prefix,bound):
    attempt,output=original.paths(c,entry,prefix)
    if not attempt.exists() and not output.exists():return 'new'
    guard=attempt/'guard_receipt.json'
    if not guard.exists():raise ValueError('Partial original fragment preserved;never rerun it')
    g=read(guard)
    if not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted']:
        raise ValueError('Original failed fragment preserved;separate resource recovery only')
    if prefix!='summary':
        _,p=original.checked_fragment(c,entry,prefix)
        entity=next(e for e in bound['dataset']['entities'] if e['array_prefix']==prefix)
        if p['source_lineage']!=bound['lineage'] or p['entity']!=entity['entity_id']:raise ValueError('Exact original grouped entity lineage required')
    else:
        path=output/'complete_case_TAB_metric_controls.json';p=read(path)
        if (not p['passed'] or p['case']!=entry['case'] or p['configuration_sha256']!=sha(ROOT/original.CONFIG)
                or p['source_lineage']!=bound['lineage'] or p['entities']!=len(bound['dataset']['entities'])
                or g['proof_sha256']!=sha(path) or g['exact_child_handle']['command']!=original.worker_command(c,entry,prefix)):
            raise ValueError('Actual complete original full-entity summary and exact guard required')
        expected=[]
        for entity in bound['dataset']['entities']:
            path,proof=original.checked_fragment(c,entry,entity['array_prefix'])
            if proof['source_lineage']!=bound['lineage']:raise ValueError('All summary fragments must share source lineage')
            expected.append(dict(path=str(path),sha256=sha(path)))
        if p['fragments']!=expected:raise ValueError('Every complete original summary fragment required')
    return 'reuse_guarded'


def stage(c,entry,prefix,api,update):
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    import psutil
    attempt,output=original.paths(c,entry,prefix)
    if attempt.exists() or output.exists():raise FileExistsError('One exact original fragment attempt only')
    attempt.mkdir(parents=True,exist_ok=False);command=original.worker_command(c,entry,prefix)
    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
        write(attempt/'launch.json',handle);update(state='running_original_TAB_metric_suite',active_pid=child.pid,
            active_case=entry['case']['id'],active_entity_prefix=prefix,exact_child_handle=handle)
        code,usage=capped_wait(child,c,api,lambda v:update(v))
    path=output/('complete_case_TAB_metric_controls.json' if prefix=='summary' else original.REPORT)
    passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
        proof_sha256=sha(path) if path.exists() else None))
    if not passed:raise ValueError('Original TAB fragment failed;preserve without retry')


def controller(config,c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing fair continuation owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every previous continuation')
    oldlock=None;mirror=False;actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        active_pid=None,required_cases=878,required_label_views=2057,all_original_experiments_complete=False,
        scheduler_configuration=CONFIG,scheduler_configuration_sha256=sha(ROOT/CONFIG))
    done=[];failed={};source_failed={};entity_done={}
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_case_keys=list(done),failed_cases_preserved=dict(failed),
            source_failures_preserved=dict(source_failed),completed_entities=dict(entity_done),captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json',state)
        if mirror:write(ROOT/c['state_root']/'status.json',state)
    def yield_slot():
        update(state='yielding_audit_lock_to_other_original_waiters',active_pid=None)
        time.sleep(config['inter_entity_yield_seconds'])
    try:
        update(state='waiting_original_entity_completion_for_safe_handoff')
        old,oldlock=handoff(config,c,api,base,update)
        keys={e['key'] for e in c['entries']}
        if not (set(old['completed_case_keys'])|set(old['failed_cases_preserved'])|set(old['source_failures_preserved']))<=keys:
            raise ValueError('Original outcome identities must belong to full878 roster')
        done.extend(old['completed_case_keys']);failed.update(old['failed_cases_preserved']);source_failed.update(old['source_failures_preserved'])
        entity_done.update(old['completed_entities'])
        for entry in c['entries']:
            if entry['key'] in done:
                bound=original.ready_case(c,entry)
                if bound is None or fragment_action(c,entry,'summary',bound)!='reuse_guarded':raise ValueError('Previously complete case requires current full guard chain')
        mirror=True;update(state='original_outcomes_retained_before_fair_continuation',active_pid=None)
        yield_slot()
        while len(done)+len(failed)+len(source_failed)<len(c['entries']):
            for entry in c['entries']:
                key=entry['key']
                if key in done or key in failed or key in source_failed:continue
                try:
                    source=original.source_status(entry)
                    if source is not None and source['kind']=='source_failed_preserved':source_failed[key]=source;continue
                    bound=original.ready_case(c,entry)
                    if bound is None:continue
                    completed=0
                    for entity in bound['dataset']['entities']:
                        prefix=entity['array_prefix'];action=fragment_action(c,entry,prefix,bound)
                        if action=='new':
                            with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],lambda v:update(v)):
                                verify(read(ROOT/CONFIG));stage(c,entry,prefix,api,update)
                            fragment_action(c,entry,prefix,bound)
                        completed+=1;entity_done[key]=completed;update(state='between_original_entities',active_pid=None)
                        if action=='new':yield_slot()
                    action=fragment_action(c,entry,'summary',bound)
                    if action=='new':
                        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],lambda v:update(v)):
                            verify(read(ROOT/CONFIG));stage(c,entry,'summary',api,update)
                        fragment_action(c,entry,'summary',bound)
                    done.append(key);update(state='between_complete_TAB_cases',active_pid=None);yield_slot()
                except Exception:failed[key]=traceback.format_exc();update(state='original_TAB_failure_preserved',active_pid=None)
            update(state='waiting_actual_native_and_fresh_all_label_reviews',active_pid=None);time.sleep(10)
        update(state='all878_metric_outcomes_terminal_other_research_requirements_remain',active_pid=None)
    finally:
        if oldlock is not None:oldlock.close()
        owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,all_original_cases=878,labels=2057)));return
    config=read(ROOT/CONFIG);c=verify(config)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=878,label_views=2057,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    else:controller(config,c)


if __name__=='__main__':main()
