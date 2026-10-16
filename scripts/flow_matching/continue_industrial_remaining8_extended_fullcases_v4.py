"""Retire one exact idle metadata owner and run eight qualified full repairs."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.continue_industrial_remaining8_extended_fullcases_v4'
TEST='tests/test_fm_industrial_remaining8_extended_fullcases_v4.py'
CONFIG='configs/runtime/fm_industrial_remaining8_extended_fullcases.v4.json'
STATE='experiments/runs/fm_industrial_remaining8_extended_fullcases_v4'
QUEUE='configs/experiments/fm_industrial_empty_output_recovery_queue.v1.json'
PACKET='projects/flow_matching_research/results/2026-10-16/industrial_remaining8_all80_real_couplings_qualified_v1'
OLD_STATE='experiments/runs/fm_industrial_empty_output_recovery_v1/status.json'
OLD_WORKER_LOCK='experiments/runs/fm_industrial_empty_output_recovery_v1/worker.lock'
OLD_CONTROLLER_LOCK='experiments/runs/fm_industrial_cached_namespace_handoff_v4/controller.lock'
EXPECTED=dict(pid=30860,create_time=1792021631.410889,command=[
    r'C:\Users\10175\AppData\Local\Programs\Python\Python312\python.exe','-B','-X','utf8','-u','-m',
    'scripts.flow_matching.run_industrial_cached_entry_handoff_v4','--config','configs/runtime/fm_industrial_cached_namespace_handoff.2026-10-15_v4.json'])


def validate_job(old,new,a,b):
    ignored={'id','output_directory','model_config','frozen_files'}
    p=dict(b['parameters']);cap=p.pop('sinkhorn_max_final_iterations',None)
    if (p!=a['parameters'] or cap!=200000 or b['entrypoint']!='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector'
            or {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in new.items() if k not in ignored}
            or new['id']!=old['id']+'__transport_extension_v4'
            or new['output_directory']!=STATE+'/jobs/'+new['id']+'/source'
            or new['frozen_files'][:len(old['frozen_files'])]!=old['frozen_files']):
        raise ValueError('Only explicit same-objective solver cap and separate repaired identity/output may differ')


def registration():
    q=read(ROOT/QUEUE);bindings=read(ROOT/PACKET/'all3_full_sequence_proof_bindings.json')['groups']
    from scripts.flow_matching.diagnose_industrial_remaining8_entropic_couplings_v1 import CONFIG as DIAGNOSTIC
    diagnostic=read(ROOT/DIAGNOSTIC);cases=[]
    for case in q['jobs'][1:]:
        old=case['original_job'];a=read(ROOT/old['model_config']);p=a['parameters'];kind='sf2m_flow_only' if p['objective']=='sf2m' and p.get('score_weight',1.)==0 else p['objective']
        model=f'configs/models/fm_objective_{kind}_stable_extended_v3.json';b=read(ROOT/model)
        job=copy.deepcopy(old);job.update(id=old['id']+'__transport_extension_v4',model_config=model)
        job['output_directory']=STATE+'/jobs/'+job['id']+'/source'
        job['frozen_files']+=[dict(path=x,sha256=sha(ROOT/x)) for x in [model,'src/iia_benchmark/models/fm_objective_stable_extended.py']]
        validate_job(old,job,a,b)
        key=old['dataset']+'__seed'+str(old['seed']);group=diagnostic['groups'][key]
        cases.append(dict(canonical_original_id=old['id'],job=job,evaluation=case['evaluation'],
            coupling_group=key,coupling_report=bindings[key]['report_path'],coupling_report_sha256=bindings[key]['report_sha256'],
            fit_windows=group['fit_windows'],optimizer_updates=group['expected_coupling_problems'],
            native_output='F:/aicoding/IIA_Data/experiments/industrial_remaining8_extended_fullcases_v4/'+job['id']+'/replay'))
    runtime=read(ROOT/'configs/runtime/fm_industrial_SB_transport_extension.v3.json')
    first=q['jobs'][0];old_failure=first['job']['output_directory']+'/failure.json'
    pins=[MODULE.replace('.','/')+'.py',TEST,'tests/test_fm_industrial_remaining8_fullcase_pipeline_v4.py','scripts/flow_matching/industrial_extended_fullcase_common_v4.py',
        QUEUE,DIAGNOSTIC,PACKET+'/packet_integrity.json',PACKET+'/all3_full_sequence_proof_bindings.json',old_failure,
        'configs/runtime/fm_industrial_SB_transport_extension.v3.json',
        'configs/runtime/fm_industrial_cached_namespace_handoff.2026-10-15_v4.json',
        'scripts/flow_matching/run_industrial_tsad_experiment.py','scripts/flow_matching/audit_industrial_SB_extended_full_replay_v1.py',
        'scripts/flow_matching/diagnose_industrial_remaining8_entropic_couplings_v1.py']
    pins+=sorted({c['job']['model_config'] for c in cases})
    return dict(previous_owner=EXPECTED,previous_state_path=OLD_STATE,previous_worker_lock=OLD_WORKER_LOCK,
        state_root=STATE,cases=cases,original_case_count=9,remaining_case_count=8,
        delegated_first_canonical_id=first['original_id'],delegated_first_runtime='configs/runtime/fm_industrial_SB_transport_extension.v3.json',
        delegated_first_source_and_native_controllers_preserved=True,settings=runtime['settings'],resource_lock=runtime['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(pins))],
        original_global_scope=runtime['original_global_scope'],original_CPU_scope=550,original_GPU_scope=850,
        full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('All eight original cases, source budgets, numerical qualification and guards must remain')
    from scripts.flow_matching import diagnose_industrial_remaining8_entropic_couplings_v1 as diagnostic
    diagnostic.verify(read(ROOT/diagnostic.CONFIG))
    for b in read(ROOT/PACKET/'all3_full_sequence_proof_bindings.json')['groups'].values():
        if sha(b['report_path'])!=b['report_sha256'] or sha(b['guard_path'])!=b['guard_sha256']:raise ValueError('Full qualified real sequence changed')
        proof,guard=read(b['report_path']),read(b['guard_path'])
        if not proof['passed'] or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']:raise ValueError('Successful full mathematical qualification required')
    if len(c['cases'])!=8 or len({x['canonical_original_id'] for x in c['cases']})!=8 or any(x['canonical_original_id']==c['delegated_first_canonical_id'] for x in c['cases']):
        raise ValueError('No omitted or duplicate original cases; first case remains delegated')
    for case in c['cases']:
        for pin in case['job']['frozen_files']:
            if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen original experiment changed: '+pin['path'])


def legacy_layout():
    q=read(ROOT/QUEUE);first=q['jobs'][0]
    for case in q['jobs']:
        path=ROOT/case['original_job']['output_directory']
        if not path.is_dir() or any(path.iterdir()):raise ValueError('Preserve every original empty source directory')
    for case in q['jobs'][1:]:
        if (ROOT/case['job']['output_directory']).exists():raise ValueError('A remaining legacy numerical attempt already started')
    failed=ROOT/first['job']['output_directory']
    if set(p.name for p in failed.iterdir())!={'failure.json','resource_usage.json'}:raise ValueError('First old failed attempt changed')
    f=read(failed/'failure.json')
    if f['exit_code']!=1 or f['resources']['resource_guard_aborted']:raise ValueError('Preserve the diagnosed numerical failure')
    return {p.name:sha(p) for p in failed.iterdir()}


def idle_allowed(expected,identity,state,children,locks):
    required={str((ROOT/p).resolve()).lower() for p in [OLD_CONTROLLER_LOCK,OLD_WORKER_LOCK]}
    first=read(ROOT/QUEUE)['jobs'][0]['job']['id']
    return (identity==expected and state.get('pid')==expected['pid'] and state.get('active_pid') is None
        and state.get('active_job') is None and state.get('shared_lock_owned') is False
        and state.get('state') in ['waiting_for_memory','waiting_for_heavy_lock']
        and state.get('jobs')=={first:'partial_preserved'} and set(locks)==required
        and all(p['name'].lower()=='conhost.exe' and str(Path(p['exe']).resolve()).lower()==str(Path('C:/Windows/System32/conhost.exe').resolve()).lower() for p in children))


def idle_snapshot(c):
    p=psutil.Process(c['previous_owner']['pid'])
    identity=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())
    raw=(ROOT/OLD_STATE).read_bytes();state=json.loads(raw)
    children=[dict(pid=x.pid,create_time=x.create_time(),name=x.name(),exe=x.exe(),command=x.cmdline()) for x in p.children(recursive=True)]
    locks=[str(Path(f.path).resolve()).lower() for f in p.open_files() if f.path.endswith('.lock')]
    if not p.is_running() or not idle_allowed(c['previous_owner'],identity,state,children,locks):raise ValueError('Only exact idle metadata owner, own locks and no numerical child may be handed off')
    ids={x['canonical_original_id'] for x in c['cases']}|{x['job']['id'] for x in c['cases']}
    ids|={x['job']['id'] for x in read(ROOT/QUEUE)['jobs'][1:]}
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(a in ids for a in process.info['cmdline'] or []):raise ValueError('A matching original or repaired numerical command already exists')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    for case in c['cases']:
        if (ROOT/case['job']['output_directory']).exists() or Path(case['native_output']).exists():raise ValueError('A repaired numerical output already exists')
    return dict(identity=identity,state=state,children=children,locks=locks,preserved_legacy_failure_sha256=legacy_layout()),raw


def handoff(c):
    base=ROOT/STATE
    first,raw=idle_snapshot(c);time.sleep(.2);second,raw=idle_snapshot(c)
    if first['identity']!=second['identity'] or first['preserved_legacy_failure_sha256']!=second['preserved_legacy_failure_sha256']:
        raise ValueError('Previous owner or failure changed between observations')
    (base/'previous_idle_status.json').write_bytes(raw);write(base/'two_pre_handoff_observations.json',dict(first=first,second=second))
    p=psutil.Process(c['previous_owner']['pid']);idle_snapshot(c)
    # Suspend only this proven metadata parent to close the last child-spawn race.
    p.suspend()
    try:
        paused,_=idle_snapshot(c);write(base/'suspended_metadata_recheck.json',paused)
    except BaseException:
        p.resume();raise
    try:p.terminate();p.wait(timeout=10)
    except BaseException:
        if p.is_running():p.resume()
        raise
    legacy_layout()
    write(base/'handoff.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),previous_owner=c['previous_owner'],
        only_exact_idle_metadata_parent_suspended_rechecked_and_retired=True,numerical_workers_stopped=False,
        first_diagnosed_failure_and_all_unstarted_eight_original_attempts_preserved=True,first_delegated_case_and_controller_preserved=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--case');parser.add_argument('--phase',choices=['source','replay']);args=parser.parse_args()
    if args.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if bool(args.case)!=bool(args.phase):raise ValueError('Case and phase must be supplied together')
    if args.probe_only:
        print(json.dumps(dict(passed=True,remaining_full_cases=8,delegated_first_case_preserved=True,full_training_commit_GiB=c['settings']['minimum_commit_headroom_bytes']/2**30,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,new_numerical_workers=0)));return
    if args.case:
        case=next(x for x in c['cases'] if x['canonical_original_id']==args.case)
        from scripts.flow_matching import industrial_extended_fullcase_common_v4 as common
        if args.phase=='source':common.train(case)
        elif args.phase=='replay':common.replay(case,ROOT/STATE/'attempts'/args.case/'source/receipt.json',case['native_output'])
        else:raise ValueError('Explicit source or replay phase required')
        return
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/STATE;base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():raise RuntimeError('Preserve existing controller and terminal attempts')
    legacy=None
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,jobs={},required_full_cases=8,
        delegated_first_case=c['delegated_first_canonical_id'],all_original_experiments_complete=False)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='running_full_original_'+state.get('phase','case')
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def run(case,phase,proof):
        cid=case['canonical_original_id'];attempt=base/'attempts'/cid/phase
        if attempt.exists():raise RuntimeError('Preserve existing numerical attempt; no automatic retry')
        update(dict(phase=phase,active_case=cid,active_pid=None))
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            attempt.mkdir(parents=True)
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--case',cid,'--phase',phase]
            with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='running_full_original_'+phase,active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,c,api,update)
            passed=code==0 and not usage['resource_guard_aborted'] and Path(proof).exists() and read(proof)['passed']
            write(attempt/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if Path(proof).exists() else None))
            update(dict(active_pid=None));return passed
    try:
        handoff(c);legacy=api['exclusive_lock'](ROOT/OLD_WORKER_LOCK)
        if legacy is None:raise RuntimeError('Legacy queue ownership changed; do not launch a numerical replacement')
        for case in c['cases']:
            cid=case['canonical_original_id'];source=ROOT/case['job']['output_directory']/'complete_training_solver_audit.json'
            if not run(case,'source',source):state['jobs'][cid]='source_failure_preserved';continue
            state['jobs'][cid]='source_complete_replay_required'
            if not run(case,'replay',Path(case['native_output'])/'report.json'):state['jobs'][cid]='replay_failure_preserved';continue
            state['jobs'][cid]='complete_source_native_and_independent_strict_metrics'
            update(dict(state='case_complete',active_pid=None))
        update(dict(state='all_eight_attempts_terminal',active_pid=None))
    except Exception as error:
        update(dict(state='terminal_controller_failure_preserved',active_pid=None,error=str(error)));raise
    finally:
        if legacy is not None:legacy.close()
        owner.close()


if __name__=='__main__':main()
