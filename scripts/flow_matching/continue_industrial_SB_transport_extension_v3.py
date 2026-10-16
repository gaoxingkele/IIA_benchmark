"""One diagnosed solver repair, full original industrial budget, no auto retry."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType
import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.continue_industrial_SB_transport_extension_v3'
CONFIG='configs/runtime/fm_industrial_SB_transport_extension.v3.json'
DIAGNOSTIC='scripts.flow_matching.diagnose_industrial_SB_transport_extension_v3'
ORIGINAL_QUEUE='configs/experiments/fm_industrial_empty_output_recovery_queue.v1.json'
MODEL='configs/models/fm_objective_sb_stable_extended_v3.json'
REPORT='F:/aicoding/IIA_Data/experiments/industrial_SB_transport_extension_v3/all20_real_coupling_problems.json'
CANONICAL='fm_objective_sb__stable_numerics_v2__industrial_v1__pronto_day4__seed1103'
STATE='experiments/runs/fm_industrial_SB_transport_extension_v3'


def validate_case(old,new,old_model,new_model):
    excluded={'id','output_directory','model_config','frozen_files'}
    if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in new.items() if k not in excluded}:
        raise ValueError('Keep complete dataset/window/seed/device/role and all original experiment fields')
    params=dict(new_model['parameters']);cap=params.pop('sinkhorn_max_final_iterations',None)
    if params!=old_model['parameters'] or cap!=200000:raise ValueError('Only the final solver cap may extend; all training/scoring parameters stay exact')
    if new_model['entrypoint']!='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector':
        raise ValueError('Use only the registered private same-objective solver extension')
    if new['id']!=old['id']+'__transport_extension_v3' or new['output_directory']!=STATE+'/full_job':raise ValueError('Separate explicit recovered identity/output required')
    if new['frozen_files'][:len(old['frozen_files'])]!=old['frozen_files']:raise ValueError('Every original frozen file must remain pinned')


def registration():
    q=read(ROOT/ORIGINAL_QUEUE);case=q['jobs'][0];assert case['original_id']==CANONICAL
    settings={k:q['settings'][k] for k in ['minimum_available_memory_bytes','minimum_commit_headroom_bytes','emergency_commit_headroom_bytes']}
    assert settings['minimum_commit_headroom_bytes']==32*2**30 and settings['emergency_commit_headroom_bytes']==16*2**30
    settings.update(maximum_worker_rss_bytes=2*2**30,maximum_worker_private_bytes=4*2**30)
    diagnostic=dict(settings,minimum_commit_headroom_bytes=22*2**30,maximum_worker_private_bytes=2*2**30)
    pins=[MODULE.replace('.','/')+'.py',DIAGNOSTIC.replace('.','/')+'.py','tests/test_fm_industrial_SB_transport_extension_v3.py',
        'src/iia_benchmark/models/fm_objective_stable_extended.py','tests/test_fm_objective_stable_extended.py',
        'src/iia_benchmark/models/fm_objective_stable.py','scripts/flow_matching/run_industrial_tsad_experiment.py',ORIGINAL_QUEUE,
        'scripts/flow_matching/cached_resource_snapshot_v1.py','scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    failure=ROOT/case['job']['output_directory']/'failure.json'
    stderr=ROOT/q['settings']['output_root']/'logs'/case['job']['id']/'stderr.log'
    pins += [failure.relative_to(ROOT).as_posix(),stderr.relative_to(ROOT).as_posix()]
    pins += [f'configs/models/fm_objective_{name}_stable_extended_v3.json' for name in ['sb','sf2m','sf2m_flow_only']]
    return dict(original_case=CANONICAL,original_queue=ORIGINAL_QUEUE,original_full_industrial_scope_CPU=550,original_full_industrial_scope_GPU=850,
        original_global_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        python='C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe',settings=settings,diagnostic_settings=diagnostic,
        diagnostic_is_coupling_only_no_optimizer_or_neural_forward=True,diagnostic_post_import_commit_requirement_bytes=20*2**30,
        resource_lock='experiments/runs/fm_heavy_recovery_cpu.lock',state_root=STATE,diagnostic_report=REPORT,
        source_receipts=[dict(path=x,sha256=sha(ROOT/x)) for x in pins],
        original_failure_sha256=sha(failure),original_stderr_sha256=sha(stderr),
        literature_citation='https://arxiv.org/abs/2302.00482',data_citation='configs/datasets/pronto_public.json',
        no_original_source_or_failed_output_overwrite=True,full_original_case_complete=False,all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Preserve full scope, all guards, failed original and unchanged source bindings')
    case=read(ROOT/ORIGINAL_QUEUE)['jobs'][0]
    failed=read(ROOT/case['job']['output_directory']/'failure.json')
    if failed['exit_code']!=1 or failed['resources']['resource_guard_aborted']:raise ValueError('This is a diagnosed numerical failure only')
    if (ROOT/case['job']['output_directory']/'model.pt').exists():raise ValueError('The old failed full attempt must remain unchanged')
    for p in psutil.process_iter(['cmdline']):
        try:
            args=p.info['cmdline'] or []
            if case['job']['id'] in args or CANONICAL in args:raise ValueError('A matching original numerical child still runs')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return case


def full_job_registration(c,case):
    r=read(REPORT)
    if (not r['passed'] or len(r['coupling_problems'])!=20 or r['new_neural_forwards'] or r['new_optimizer_updates']
            or not any(x['used_extended_cap'] for x in r['coupling_problems'])
            or any(x['actual_absolute_marginal_error']>1e-6 or x['regularization']!=2. for x in r['coupling_problems'])):
        raise ValueError('First qualify all20 actual coupling problems at the original epsilon/tolerance')
    original=case['original_job'];job=copy.deepcopy(original)
    job.update(id=original['id']+'__transport_extension_v3',output_directory=STATE+'/full_job',model_config=MODEL)
    extras=[MODEL,'src/iia_benchmark/models/fm_objective_stable_extended.py']
    job['frozen_files'] += [dict(path=x,sha256=sha(ROOT/x)) for x in extras]
    validate_case(original,job,read(ROOT/original['model_config']),read(ROOT/MODEL))
    return dict(job=job,evaluation=case['evaluation'],canonical_original_case=original['id'],coupling_diagnostic_sha256=sha(REPORT),
        source_runtime=CONFIG,source_runtime_sha256=sha(ROOT/CONFIG),full_original_epochs=20,full_original_fit_windows=8,
        full_original_test_points=11700,full_original_validation_points=4320,numerical_history_bitwise_equivalence_claim=False)


def worker(c):
    from scripts.flow_matching import run_industrial_tsad_experiment as frozen
    case=verify(c);registration=read(ROOT/STATE/'full_original_case_job.json')
    if registration!=full_job_registration(c,case):raise ValueError('Full original job registration changed')
    instances=[];builder=frozen.build_configured_method
    def observe(*args,**kwargs):
        result=builder(*args,**kwargs);instances.append(result);return result
    body=frozen.run
    private=FunctionType(body.__code__,dict(body.__globals__,build_configured_method=observe),body.__name__,body.__defaults__,body.__closure__)
    result=private(ROOT,registration['job'],registration['evaluation'])
    assert len(instances)==1 and len(instances[0].training_losses_)==20
    audits=instances[0].transport_solver_audit_
    assert len(audits)==20 and all(x['actual_absolute_marginal_error']<=1e-6 and x['regularization']==2. for x in audits)
    assert result['test_points']==11700 and result['validation_points']==4320 and result['unique_fit_windows']==8
    write(ROOT/STATE/'full_job/complete_transport_solver_audit.json',dict(passed=True,all20_actual_training_couplings=audits,
        source_training_and_scoring_function_code_object_unchanged=True,original_epochs=20,original_windows=8,optimizer_updates=20,
        final_regularization=2.,absolute_marginal_tolerance=1e-6,source_result_sha256=sha(ROOT/STATE/'full_job/result.json'),
        current_local_adaptation_not_historical_paper_or_full_TAB_certification=True))


def controller(c):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/STATE;base.mkdir(parents=True,exist_ok=True)
    handle=api['exclusive_lock'](base/'controller.lock')
    if handle is None:raise RuntimeError('This single diagnosed-case successor is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,canonical_case=CANONICAL,
        full_training_complete=False,all_original_experiments_complete=False,configuration_sha256=sha(ROOT/CONFIG))
    def update(snapshot=None,**kw):
        state.update(snapshot or {});state.update(kw);state['captured_utc']=datetime.now(timezone.utc).isoformat();write(base/'status.json',state)
    def run(stage,settings,command,proof):
        attempt=base/'attempts'/stage
        if attempt.exists():raise RuntimeError('Preserve any previous attempt; no automatic retry')
        with api['resource_slot'](ROOT/c['resource_lock'],settings,lambda v:update(v,active_pid=None,stage=stage),poll_seconds=10):
            attempt.mkdir(parents=True)
            with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                p=psutil.Process(child.pid);update(state='running_'+stage,stage=stage,active_pid=p.pid,active_create_time=p.create_time(),actual_command=command)
                code,usage=capped_wait(child,dict(settings=settings),api,lambda v:update(dict(v,state='running_'+stage),stage=stage))
            passed=code==0 and not usage['resource_guard_aborted'] and Path(proof).exists() and read(proof).get('passed') is True
            write(attempt/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if Path(proof).exists() else None))
            if not passed:raise RuntimeError('Preserved terminal '+stage+' failure; no automatic retry')
    try:
        case=verify(c);update(state='waiting_for_real_coupling_diagnostic',stage='coupling_diagnostic')
        run('coupling_diagnostic',c['diagnostic_settings'],[c['python'],'-B','-X','utf8','-u','-m',DIAGNOSTIC],REPORT)
        job=full_job_registration(c,case)
        with (base/'full_original_case_job.json').open('x',encoding='utf-8') as f:json.dump(job,f,indent=2)
        update(state='waiting_original_full_training_resources',active_pid=None,real_coupling_diagnostic_complete=True)
        run('full_training',c['settings'],[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker'],ROOT/STATE/'full_job/complete_transport_solver_audit.json')
        update(state='complete_source_training_and_scores_independent_replay_still_required',active_pid=None,full_training_complete=True)
    except Exception as error:
        update(state='terminal_failure_preserved',active_pid=None,error=str(error));raise
    finally:handle.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,full_training_commit_GiB=c['settings']['minimum_commit_headroom_bytes']/2**30,numerical_workers_started=0)))
    elif a.worker:worker(c)
    else:controller(c)


if __name__=='__main__':main()
