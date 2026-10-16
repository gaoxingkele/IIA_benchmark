"""Separate bounded math diagnostics from the unchanged full-training CPU lock."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from types import FunctionType
import psutil
from scripts.flow_matching import continue_industrial_SB_transport_extension_v3 as source
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.continue_industrial_SB_math_diagnostic_lock_v4'
CONFIG='configs/runtime/fm_industrial_SB_math_diagnostic_lock.v4.json'
TEST='tests/test_fm_industrial_SB_math_diagnostic_lock_v4.py'
BASE='experiments/runs/fm_industrial_SB_math_diagnostic_lock_v4'
PREVIOUS=dict(pid=24480,create_time=1792156834.6153083,command=[
    'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe','-B','-X','utf8','-u','-m',source.MODULE])
DIAGNOSTIC_LOCK='experiments/runs/fm_bounded_transport_math_diagnostic_cpu.lock'


def selected_lock(runtime,settings):
    if settings==runtime['diagnostic_settings']:return DIAGNOSTIC_LOCK
    if settings==runtime['settings']:return runtime['resource_lock']
    raise ValueError('Only the original exact diagnostic or full training guard may be used')


def original_controller_with_diagnostic_lock():
    path=ROOT/(source.MODULE.replace('.','/')+'.py')
    tree=ast.parse(path.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    target=ast.dump(ast.parse("ROOT / c['resource_lock']",mode='eval').body)
    changes=[]
    class LockSelection(ast.NodeTransformer):
        def visit_BinOp(self,item):
            if ast.dump(item)==target:
                changes.append(item)
                return ast.copy_location(ast.parse('ROOT / _selected_stage_lock(c, settings)',mode='eval').body,item)
            return self.generic_visit(item)
    node=LockSelection().visit(node)
    if len(changes)!=1:raise ValueError('Change exactly one controller resource-lock path expression')
    unit=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace=dict(source.controller.__globals__,_selected_stage_lock=selected_lock)
    exec(compile(unit,str(path),'exec'),namespace)
    entry=namespace['controller'];entry.scheduler_source_ast=unit
    return entry


def registration():
    c=read(ROOT/source.CONFIG);assert c==source.registration()
    pins=[MODULE.replace('.','/')+'.py',TEST,source.CONFIG,source.MODULE.replace('.','/')+'.py',source.DIAGNOSTIC.replace('.','/')+'.py',
        'src/iia_benchmark/models/fm_objective_stable_extended.py']
    return dict(original_runtime=source.CONFIG,original_runtime_sha256=sha(ROOT/source.CONFIG),previous_owner=PREVIOUS,
        diagnostic_resource_lock=DIAGNOSTIC_LOCK,full_training_resource_lock=c['resource_lock'],
        diagnostic_settings=c['diagnostic_settings'],full_training_settings=c['settings'],handoff_state_root=BASE,
        original_global_scope=c['original_global_scope'],original_canonical_case=source.CANONICAL,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        numerical_commands_or_bodies_changed=False,original_full_training_gates_or_lock_changed=False,
        diagnostic_runs_no_neural_forward_or_optimizer=True,all_original_experiments_complete=False)


def idle_snapshot(expected):
    p=psutil.Process(expected['pid'])
    if not p.is_running() or p.create_time()!=expected['create_time'] or p.cmdline()!=expected['command']:raise ValueError('Exact metadata owner identity changed')
    children=[x for x in p.children(recursive=True) if x.name().lower()=='python.exe']
    if children:raise ValueError('Numerical child exists; no metadata handoff')
    base=ROOT/source.STATE;raw=(base/'status.json').read_bytes();s=json.loads(raw)
    if (s.get('pid')!=expected['pid'] or s.get('create_time')!=expected['create_time'] or s.get('active_pid') is not None
            or s.get('stage')!='coupling_diagnostic' or s.get('state') not in ['waiting_for_memory','waiting_for_heavy_lock']
            or s.get('shared_lock_owned') is not False or s.get('full_training_complete')):
        raise ValueError('Only the idle pre-diagnostic metadata controller may be handed off')
    if (base/'attempts').exists() or (base/'full_original_case_job.json').exists() or Path(source.REPORT).exists():raise ValueError('A numerical attempt already started; preserve it')
    locks={str(Path(x.path).resolve()).lower() for x in p.open_files() if x.path.endswith('.lock')}
    own=str((base/'controller.lock').resolve()).lower()
    if own not in locks or str((ROOT/read(ROOT/source.CONFIG)['resource_lock']).resolve()).lower() in locks:
        raise ValueError('Exact controller lane ownership without shared training-lock ownership required')
    return dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),no_numerical_descendants=True,
        pre_numerical_outputs_absent=True,owns_only_metadata_lane=True,state=s),raw


def handoff(c):
    base=ROOT/BASE;base.mkdir(parents=True,exist_ok=True)
    if (base/'handoff.json').exists():
        proof=read(base/'handoff.json')
        if proof['previous_owner']!=c['previous_owner'] or not proof['only_idle_metadata_owner_stopped']:raise ValueError('Prior handoff differs')
        return
    first,raw=idle_snapshot(c['previous_owner']);time.sleep(.1);second,second_raw=idle_snapshot(c['previous_owner'])
    for k in ['pid','create_time','command','no_numerical_descendants','pre_numerical_outputs_absent','owns_only_metadata_lane']:
        if first[k]!=second[k]:raise ValueError('Owner changed between independent observations')
    (base/'previous_idle_status.json').write_bytes(second_raw)
    write(base/'two_pre_handoff_observations.json',dict(first=first,second=second))
    # Recheck immediately before stopping only our own known idle metadata PID.
    idle_snapshot(c['previous_owner'])
    p=psutil.Process(c['previous_owner']['pid']);p.terminate();p.wait(timeout=10)
    if (ROOT/source.STATE/'attempts').exists():raise RuntimeError('A concurrent numeric attempt appeared; preserve it and do not launch a replacement')
    write(base/'handoff.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),previous_owner=c['previous_owner'],
        only_idle_metadata_owner_stopped=True,numerical_processes_stopped=False,original_runtime_sha256=c['original_runtime_sha256'],
        full_training_lock_and_all_original_guards_unchanged=True,diagnostic_lock=DIAGNOSTIC_LOCK))


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Full original scope, exact guarded handoff and scheduling-only source pins required')
    runtime=read(ROOT/source.CONFIG);source.verify(runtime)
    entry=original_controller_with_diagnostic_lock()
    if a.probe_only:
        print(json.dumps(dict(passed=True,full_training_lock=selected_lock(runtime,runtime['settings']),
            diagnostic_lock=selected_lock(runtime,runtime['diagnostic_settings']),new_numerical_workers=0)));return
    handoff(c);entry(runtime)


if __name__=='__main__':main()
