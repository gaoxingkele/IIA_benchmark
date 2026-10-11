import ast
import copy
from types import SimpleNamespace

import pytest

from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import ROOT, capped_class_ast, validate_limits
from scripts.flow_matching.handoff_bounded_strict_cpu_v2 import verify_idle
from scripts.flow_matching import run_resource_gated_light_controller_v1 as original


def limits():
    before=dict(minimum_available_memory_bytes=12,minimum_commit_headroom_bytes=32,emergency_commit_headroom_bytes=16,
                guard_poll_seconds=2,controller_yield_seconds=2)
    after=dict(before,minimum_available_memory_bytes=16,maximum_worker_rss_bytes=10,maximum_worker_private_bytes=12)
    return before,after


def test_no_original_memory_reserve_can_decrease():
    before,after=limits();validate_limits(before,after)
    for field in ('minimum_available_memory_bytes','minimum_commit_headroom_bytes','emergency_commit_headroom_bytes'):
        wrong=dict(after);wrong[field]=before[field]-1
        with pytest.raises(ValueError,match='safeguard'):validate_limits(before,wrong)
    wrong=dict(after,maximum_worker_private_bytes=20)
    with pytest.raises(ValueError,match='emergency reserve'):validate_limits(before,wrong)
    wrong=dict(after,guard_poll_seconds=5)
    with pytest.raises(ValueError,match='cadence'):validate_limits(before,wrong)


def test_original_guard_class_only_adds_rss_and_private_cap_conditions():
    source=(ROOT/'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text(encoding='utf-8')
    node=capped_class_ast(source)
    guard=[n for n in ast.walk(node) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp)
           and ast.unparse(n.test).startswith("available < self.settings['emergency_commit_headroom_bytes']")]
    assert len(guard)==1 and len(guard[0].test.values)==3
    assert 'maximum_worker_rss_bytes' in ast.unparse(guard[0].test.values[1])
    assert 'maximum_worker_private_bytes' in ast.unparse(guard[0].test.values[2])
    guard[0].test=guard[0].test.values[0]
    original_node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GuardedChild')
    assert ast.dump(node)==ast.dump(original_node)


@pytest.mark.parametrize('rss,private,commit,aborted',[(7,9,40,False),(11,11,40,True),(9,13,40,True),(7,9,15,True)])
def test_actual_child_guard_retains_commit_emergency_and_enforces_both_new_caps(monkeypatch,tmp_path,rss,private,commit,aborted):
    source=(ROOT/'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text(encoding='utf-8')
    namespace=dict(vars(original));node=capped_class_ast(source)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'test-capped-class','exec'),namespace)
    process=SimpleNamespace(pid=123,code=None)
    process.poll=lambda:process.code
    handle=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss,private=private),children=lambda recursive:[])
    monkeypatch.setattr(original.psutil,'Process',lambda pid:handle)
    stopped=[];released=[]
    def terminate(child):stopped.append(child.pid);child.code=1
    api=dict(resource_snapshot=lambda:dict(commit_available_bytes=commit),terminate_tree=terminate)
    reservation=SimpleNamespace(__exit__=lambda *args:released.append(args))
    settings=dict(limits()[1],controller_yield_seconds=0)
    child=namespace['GuardedChild'](process,reservation,api,settings,lambda v:None,tmp_path/'receipt.json',{})
    assert child.poll()==(1 if aborted else None)
    assert child.peaks['resource_guard_aborted'] is aborted
    assert stopped==([123] if aborted else [])
    if aborted:
        child.poll();assert len(released)==1


def idle_case():
    old=dict(pid=10,create_time=1,command=['python','owned-runtime'])
    registration=dict(predecessor=old,parent_resource_registration_sha256='parent',controllers={'strict_cpu':dict(queue_sha256='queue')},
                      requested_job='next',requested_command=['python','original-worker','next'])
    state=dict(pid=10,create_time=1,runtime_sha256='parent',queue_sha256='queue',state='waiting_for_heavy_lock',active_pid=None,
               active_job='previous',shared_lock_owned=False,requested_job='next',requested_command=registration['requested_command'])
    observed=dict(old,children=[dict(name='conhost.exe')])
    terminal=dict(job_id='previous',exit_code=0,resource_guard_aborted=False,full_registered_budget_retained=True,queue_sha256='queue')
    return registration,state,observed,terminal


def test_stale_completed_job_field_is_allowed_only_with_exact_terminal_proof():
    args=idle_case();before=copy.deepcopy(args);assert verify_idle(*args)['verified_idle'];assert args==before
    args[-1]['exit_code']=1
    with pytest.raises(ValueError,match='terminal proof'):verify_idle(*args)


@pytest.mark.parametrize('change',['pid_reused','active_worker','native_child','wrong_command','wrong_queue','wrong_pending'])
def test_no_live_worker_or_unowned_controller_can_be_retired(change):
    registration,state,observed,terminal=idle_case()
    if change=='pid_reused':observed['create_time']=2
    elif change=='active_worker':state['active_pid']=123
    elif change=='native_child':observed['children'].append(dict(name='python.exe'))
    elif change=='wrong_command':observed['command']=['different-process']
    elif change=='wrong_queue':state['queue_sha256']='changed'
    elif change=='wrong_pending':state['requested_job']='other'
    with pytest.raises(ValueError,match='idle controller'):verify_idle(registration,state,observed,terminal)
