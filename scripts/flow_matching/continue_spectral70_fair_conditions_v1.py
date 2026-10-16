"""Resume all70 frozen generative reexecutions, yield GPU lock between cases.

Only a proven idle metadata parent is replaced. Original worker command,
configuration, stage creation, full samplers/evaluators and guards are retained.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as original
from scripts.flow_matching import continue_TAB878_fair_entities_v1 as fair
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic

ROOT,read,sha=original.ROOT,original.read,original.sha
MODULE='scripts.flow_matching.continue_spectral70_fair_conditions_v1'
CONFIG='configs/runtime/fm_spectral70_fair_condition_continuation.v1.json'
TEST='tests/test_fm_spectral70_fair_condition_continuation_v1.py'
BASE='experiments/runs/fm_spectral70_fair_condition_continuation_v1'
FAIRNESS='configs/runtime/fm_grasp_full_gpu_memory_fairness.v6.json'
EXPECTED=dict(pid=37832,create_time=1792132776.595039,command=[
    'C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe','-B','-X','utf8','-u','-m',original.MODULE])

def registration():
    c=read(ROOT/original.CONFIG);original.verify(c)
    pins={r['path'] for r in c['source_receipts']}
    pins.update([original.CONFIG,FAIRNESS,fair.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',MODULE.replace('.','/')+'.py',TEST])
    return dict(original_configuration=original.CONFIG,original_configuration_sha256=sha(ROOT/original.CONFIG),
        state_root=BASE,previous_owner=copy.deepcopy(EXPECTED),required_conditions=70,required_cohorts=14,
        original_scope=c['original_scope'],settings=c['settings'],resource_lock=c['resource_lock'],
        spectral_flow_minimum_commit_headroom_bytes=c['spectral_flow_minimum_commit_headroom_bytes'],
        inter_condition_yield_seconds=read(ROOT/FAIRNESS)['yield_seconds'],handoff_poll_seconds=.25,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        original_scientific_state_root=c['state_root'],original_scientific_artifact_root=c['artifact_root'],
        no_partial_attempt_retry=True,all_original_experiments_complete=False,
        boundary='Same all70conditions/14five-seed groups;two full original samplers plus four original evaluators each5runs. '
            'Original worker module/config/commands/caps/solver/precision/arrays/RNG/state checks retained. '
            'Takeover only with actual shared GPU lock,twice verified exact idle parent identity,no numerical descendants, '
            'terminal exact guard and every existing attempt represented as verified or preserved failure. '
            'Rehash all successful reports/artifacts/source proofs before resume;no original successful or failed slot rerun. '
            'Yield35seconds after each newly complete condition,matching existing GRASP fairness. '
            'Original source/scientific paths preserved;old metadata raw bytes saved before atomic status mirroring. '
            'No numerical child stopped,no budgets or thresholds reduced,no added training credit;full45/681/273/702scope remains.')

def verify(config):
    if config!=registration():raise ValueError('All70conditions/14cohorts and original numerical/resource/fairness bindings required')
    c=read(ROOT/original.CONFIG);parent,cases=original.verify(c);return c,parent,cases

def boundary_allowed(state,expected,descendants,terminal,attempt_ids=()):
    verified=state.get('verified_ids',[]);failed=state.get('failed_preserved',{})
    return (state.get('pid')==expected['pid'] and state.get('create_time')==expected['create_time']
        and state.get('required_conditions')==70 and state.get('required_cohorts')==14 and terminal
        and not set(verified)&set(failed) and set(attempt_ids)<=set(verified)|set(failed)
        and all(fair.allowed_console_child(x['name'],x['exe']) for x in descendants)
        and state.get('state') in {'scanning_all70_full_source_proofs','waiting_original_full_source_and_artifact_proofs',
            'waiting_for_memory','waiting_for_heavy_lock','waiting_for_original_GPU_memory','running',
            'running_full_generation_and_all_evaluators','yielding_shared_GPU_lock_between_complete_conditions'})

def terminal_child(c,state):
    import psutil
    if state.get('active_pid') is None:return True
    identity=state.get('active_condition')
    if not identity:return False
    path=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json'
    if not path.exists():return False
    g=read(path);h=g['exact_child_handle']
    expected=[str(ROOT/c['python']),'-B','-X','utf8','-u','-m',original.MODULE,'--case',identity]
    if h['pid']!=state['active_pid'] or h['command']!=expected or type(g.get('exit_code')) is not int:return False
    try:
        if psutil.Process(h['pid']).create_time()==h['create_time']:return False
    except psutil.NoSuchProcess:pass
    return True

def restored_reports(c,parent,cases,old):
    known=set(old['verified_ids'])|set(old['failed_preserved'])
    attempts=ROOT/c['state_root']/'attempts'
    if (not known<=set(cases) or set(old['verified_ids'])&set(old['failed_preserved'])
            or any(p.name not in known for p in attempts.iterdir() if p.is_dir())):
        raise ValueError('Every original attempt must have an immutable terminal outcome;no retry')
    reports={};bindings={}
    for identity in old['verified_ids']:
        path=Path(c['artifact_root'])/identity/'full_generation_and_evaluator_reexecution.json'
        gp=attempts/identity/'guard_receipt.json';r,g=read(path),read(gp)
        if not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['report_sha256']!=sha(path):raise ValueError('Original successful full guard/report required')
        if not r['passed'] or r['canonical_id']!=identity or r['actual_new_generator_training_updates']!=0 or r['full_original_sampling_calls']!=2 or r['original_evaluation_repeats']!=5:raise ValueError('Complete original two-sampler/four-evaluator budget required')
        for field in ['original_checkpoint_unchanged','original_source_and_frozen_dataset_verified',
            'every_model_EMA_optimizer_state_unchanged','all_generated_payload_bytes_replayed',
            'complete_RNG_restoration_and_equal_consumption','original_sampler_grid_and_solver_unchanged',
            'all_four_original_evaluators_reexecuted','all_learned_evaluators_independently_retrained']:
            if not r[field]:raise ValueError('Successful original native state/RNG/full-budget proof required')
        for artifact in r['artifacts']:
            if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Original reexecution artifact changed')
        proof,proof_path=original.ready_proof(parent,cases[identity],identity)
        if r['original_artifact_proof_sha256']!=sha(proof_path):raise ValueError('Original source artifact proof changed')
        if r['original_result_sha256']!=sha(ROOT/cases[identity]['job']['output_directory']/'result.json'):raise ValueError('Original source result changed')
        if set(r['metrics'])!={'context_fid','correlational','discriminative','predictive'}:raise ValueError('All four evaluators required')
        for item in r['metrics'].values():
            values=item['raw_repeats']
            if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('All five actual evaluator runs required')
            if not math.isclose(statistics.mean(values),item['mean'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Actual evaluator means changed')
        reports[identity]=r;bindings[identity]=dict(report_path=str(path),report_sha256=sha(path),guard_path=str(gp),guard_sha256=sha(gp))
    return reports,bindings

def initial_state(fresh,old):
    value=dict(fresh);value['verified_ids']=list(old['verified_ids']);value['failed_preserved']=copy.deepcopy(old['failed_preserved'])
    value.update(scheduler_configuration=CONFIG,scheduler_configuration_sha256=sha(ROOT/CONFIG))
    return value

def controller_program(c,old,reports,oldlock,base):
    node=next(n for n in ast.parse(Path(original.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class Metadata(ast.NodeTransformer):
        def visit_Expr(self,n):
            self.generic_visit(n)
            call=n.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id in ('base','out') and call.func.attr=='mkdir':
                return ast.copy_location(ast.Expr(ast.Call(ast.Name('_existing',ast.Load()),[call.func.value],[])),n)
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='update' and any(isinstance(x,ast.Constant) and x.value=='scanning_all70_full_source_proofs' for x in ast.walk(call)):
                return [n,ast.parse('_yield_completed(identity, verified, update)').body[0]]
            return n
        def visit_Assign(self,n):
            self.generic_visit(n)
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                target=n.targets[0].id
                if target=='owner':n.value=ast.Name('_owned_lane',ast.Load())
                elif target=='verified' and isinstance(n.value,ast.Dict):n.value=ast.Name('_verified',ast.Load())
                elif target=='state' and isinstance(n.value,ast.Call):n.value=ast.Call(ast.Name('_initial',ast.Load()),[n.value],[])
            return n
    transformed=ast.fix_missing_locations(ast.Module(body=[Metadata().visit(node)],type_ignores=[]))
    def existing(path):
        if not path.is_dir():raise ValueError('Only unchanged existing scientific directories may be resumed')
    yielded=set(reports)
    def yield_completed(identity,verified,update):
        if identity in verified and identity not in yielded:
            yielded.add(identity);update(dict(state='yielding_shared_GPU_lock_between_complete_conditions',active_pid=None))
            time.sleep(read(ROOT/CONFIG)['inter_condition_yield_seconds'])
    def write(path,value):
        atomic(path,value)
        if Path(path)==ROOT/c['state_root']/'status.json':atomic(base/'status.json',value)
    ns=dict(original.controller.__globals__,_owned_lane=oldlock,_verified=reports,
        _initial=lambda fresh:initial_state(fresh,old),_existing=existing,_yield_completed=yield_completed,write=write)
    exec(compile(transformed,str(original.__file__),'exec'),ns)
    return ns['controller'],transformed

def run(config,c,parent,cases):
    import os
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Fair full70 observer already owned')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every scheduler attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        required_conditions=70,required_cohorts=14,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {},**kw,captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    lane=None
    try:
        update(state='waiting_complete_original_reexecution_for_safe_metadata_handoff')
        def boundary(*args):
            attempts=ROOT/c['state_root']/'attempts'
            return boundary_allowed(*args,attempt_ids=[p.name for p in attempts.iterdir() if p.is_dir()])
        ns=dict(fair.__dict__,boundary_allowed=boundary,terminal_child=terminal_child,write=atomic)
        old,lane=FunctionType(fair.handoff.__code__,ns)(config,c,api,base,update)
        reports,bindings=restored_reports(c,parent,cases,old)
        fair.exclusive(base/'restored_verified_report_bindings.json',dict(records=bindings,failed_preserved=old['failed_preserved'],new_numerical_completions=0))
        # Let existing GRASP/source waiters enter after takeover, before trying
        # to reacquire the same GPU lock for the next original case.
        update(state='yielding_shared_GPU_lock_after_verified_idle_handoff')
        time.sleep(config['inter_condition_yield_seconds'])
        program,_=controller_program(c,old,reports,lane,base);program(c,parent,cases)
    finally:
        if lane is not None:lane.close()
        owner.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:fair.exclusive(ROOT/CONFIG,registration());return
    config=read(ROOT/CONFIG);c,parent,cases=verify(config)
    if a.probe_only:print(json.dumps(dict(passed=True,conditions=len(cases),cohorts=14,
        original_worker_module=original.MODULE,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    else:run(config,c,parent,cases)

if __name__=='__main__':main()
