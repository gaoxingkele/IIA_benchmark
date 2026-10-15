"""Cold metadata handoff retaining the full original RCGAN238+262 execution."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v4 as entry
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as common

ROOT,read,sha=entry.ROOT,entry.read,entry.sha
MODULE='scripts.flow_matching.continue_rcgan_waiting_host_cold_v5'
CONFIG='configs/runtime/fm_rcgan_waiting_host_cold.v5.json'
PRODUCER='configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v4.json'
TEST='tests/test_fm_rcgan_waiting_host_cold_v5.py'
BASE='experiments/runs/fm_rcgan_waiting_host_cold_v5'
EXPECTED=dict(pid=28796,create_time=1791748470.4510753,command=[
    'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe','-X','utf8','-u','-m',entry.__name__,
    '--runtime-config',PRODUCER,'--config','configs/reproducibility/fm_rcgan_seed1103_preserved_full262_retry.2026-10-12_v1.json'])


def expected_binding(native,queue,state):
    return dict(configuration_sha256=sha(ROOT/native['configuration_path']),
        checkpoint_state_proof_sha256=sha(ROOT/state['output_directory']/'checkpoint_resume_state_audit.json'),
        checkpoint_state_terminal_sha256=sha(ROOT/state['state_root']/'receipt.json'),
        native_evaluator_proof_sha256=sha(ROOT/queue['native_evaluator_proof']),
        native_evaluator_terminal_sha256=sha(ROOT/queue['native_evaluator_execution_receipt']))


def registration():
    production=read(ROOT/PRODUCER);native,queue,state=entry.verify(production)
    binding=production['state_root']+'/full_native_resume_proof_binding.json'
    value=expected_binding(native,queue,state)
    if read(ROOT/binding)!=value:raise ValueError('Exact existing full native checkpoint/evaluator binding required')
    paths={MODULE.replace('.','/')+'.py',TEST,PRODUCER,binding,production['native_configuration'],
        entry.__name__.replace('.','/')+'.py','scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py'}
    paths.update(p['path'] for p in production['source_receipts'])
    return dict(producer_configuration=PRODUCER,previous_owner=deepcopy(EXPECTED),
        previous_state_path=production['state_root']+'/status.json',state_root=BASE,
        immutable_proof_binding=binding,immutable_proof_binding_sha256=sha(ROOT/binding),proof_binding=value,
        recovery_output_directory=native['recovery_job']['output_directory'],original_queue='configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json',
        required_original_seed_slots=5,original_full_epochs=500,verified_prior_epochs=238,remaining_full_epochs=262,
        original_scope=production['original_scope'],settings=production['settings'],resource_lock=production['resource_lock'],
        stronger_admission_commit_bytes=native['retry_admission_commit_headroom_bytes'],
        native_python=queue['python'],original_arguments=EXPECTED['command'][EXPECTED['command'].index('-m')+2:],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],all_original_experiments_complete=False,
        boundary='Original5seeds/full500epochs,exact238saved+262remaining and frozen Py37 worker/optimizer/RNG/evaluators/data preserved. '
            'Only idle metadata owner handed off. Original host AST changes only reuse of existing byte-identical metadata proof binding;never overwrites it. '
            'All other host/control/numeric statements,workerargv,caps/40GiBadmission/16emergency/familylock unchanged. '
            'Cached equivalent resource telemetry only;all45/681/273/702research remains.')


def verify(c):
    if c!=registration():raise ValueError('Exact full500=238+262 native recovery,immutable proof and original policies required')
    return entry.verify(read(ROOT/PRODUCER))


def host_ast(source,expected_sha):
    if len(expected_sha)!=64 or any(c not in '0123456789abcdef' for c in expected_sha):raise ValueError('Exact proof SHA256 required')
    tree=entry.host_dispatch_ast(source);matches=0
    ref=ast.dump(ast.parse('binding_path.exists()',mode='eval').body)
    for parent in ast.walk(tree):
        body=getattr(parent,'body',None)
        if not isinstance(body,list):continue
        for index,node in enumerate(list(body)):
            if not isinstance(node,ast.If) or ast.dump(node.test)!=ref:continue
            if (len(node.body)!=1 or not isinstance(node.body[0],ast.Raise) or node.orelse
                    or not isinstance(node.body[0].exc,ast.Call) or getattr(node.body[0].exc.func,'id',None)!='FileExistsError'):
                raise ValueError('Exact original immutable-binding guard required')
            following=body[index+1]
            if ast.dump(following)!=ast.dump(ast.parse('write(binding_path,binding)').body[0]):raise ValueError('Exact original proof write required')
            check=ast.parse("if read(binding_path)!=binding: raise ValueError('Preserved exact native proof binding changed')").body[0]
            check.test=ast.BoolOp(op=ast.Or(),values=[check.test,ast.Compare(left=ast.parse('sha(binding_path)',mode='eval').body,
                ops=[ast.NotEq()],comparators=[ast.Constant(expected_sha)])])
            node.body=[check];node.orelse=[following];body.remove(following);matches+=1
    if matches!=1:raise ValueError('Only one immutable metadata proof reuse may change')
    return ast.fix_missing_locations(tree)


def idle_allowed(c,state,expected,descendants):
    return (common.idle_allowed(state,expected,descendants)
        and not Path(c['recovery_output_directory']).exists()
        and sha(ROOT/c['immutable_proof_binding'])==c['immutable_proof_binding_sha256'])


def runtime_namespace(c):
    return dict(entry.__dict__,host_dispatch_ast=lambda source:host_ast(source,c['immutable_proof_binding_sha256']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:common.exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,full_epochs=500,remaining=262)));return
    c=read(ROOT/CONFIG);verify(c)
    common.install_cached_loader(lambda value:None)
    if a.probe_only:
        import psutil
        source=Path(entry.original.previous.original.__file__).read_text(encoding='utf-8')
        compile(host_ast(source,c['immutable_proof_binding_sha256']),'<unchanged-full-RCGAN-host>','exec')
        print(json.dumps(dict(passed=True,original_seed_slots=5,full_epochs=500,prior_epochs=238,remaining=262,
            numpy_imported='numpy' in sys.modules,tensorflow_imported='tensorflow' in sys.modules,private_bytes=psutil.Process().memory_info().private)));return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True);owner=resource_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing cold RCGAN host preserved')
    if (base/'successor_entry.json').exists():owner.close();raise FileExistsError('Preserve previous host continuation')
    try:
        role=dict(previous_owner=c['previous_owner'],previous_state_path=c['previous_state_path'])
        ns=dict(common.__dict__,idle_allowed=lambda state,expected,desc:idle_allowed(c,state,expected,desc))
        FunctionType(common.handoff.__code__,ns,argdefs=common.handoff.__defaults__)(role,base)
        verify(c)
        common.exclusive(base/'successor_entry.json',dict(entry=entry.__name__,arguments=c['original_arguments'],
            proof_binding_sha256=c['immutable_proof_binding_sha256'],native_worker_unchanged=True))
        argv=sys.argv
        try:
            sys.argv=[entry.__name__,*c['original_arguments']]
            FunctionType(entry.main.__code__,runtime_namespace(c))()
        finally:sys.argv=argv
    finally:owner.close()


if __name__=='__main__':main()
