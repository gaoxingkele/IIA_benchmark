import ast
import inspect
import pytest
from scripts.flow_matching import run_crossad_post_fixed_controllers_v2 as fixed


@pytest.mark.parametrize('role',['original','recovery'])
def test_all_original_model_point_jobs_guards_and_failure_handling_retained(role):
    tree=fixed.controller_ast(role);calls=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value==fixed.POST_MODULE:
            node.value='scripts.flow_matching.run_crossad_completion_replay_queue_v1';calls+=1
    assert calls==1
    if role=='recovery':
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While) and ast.unparse(n.test)=='len(covered) < 139')
        assert ast.unparse(loop.body[0])=='_adopt_external_recovery(config, jobs, routes, recovered, covered)'
        loop.body.pop(0)
    source=fixed.original if role=='original' else fixed.recovery
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(source.controller)))


@pytest.mark.parametrize('role,phase',[
    ('original','waiting_actual_original_full_execution_completion'),
    ('recovery','waiting_actual_exact_recovery_and_original_full139_completion')])
def test_only_verified_idle_controller_can_be_handed_off(role,phase):
    state=dict(state=phase,active_pid=None,verified_original_ids=['one'])
    assert fixed.allowed_idle(role,state,False,[])
    assert not fixed.allowed_idle(role,dict(state,active_pid=42),False,[])
    assert not fixed.allowed_idle(role,state,True,[])
    assert not fixed.allowed_idle(role,dict(state,state='replaying_full_checkpoints'),False,[])
    assert not fixed.allowed_idle(role,state,False,[dict(name='python.exe',exe='C:/Python/python.exe')])
    assert not fixed.allowed_idle(role,dict(state,verified_original_ids=list(range(139))),False,[])


@pytest.mark.parametrize('terminal',[
    None,dict(exit_code=1,resource_guard_aborted=False,independent_stage_passed=False),
    dict(exit_code=0,resource_guard_aborted=True,independent_stage_passed=True)])
def test_pending_failed_or_guard_aborted_external_review_cannot_add_canonical_credit(monkeypatch,terminal):
    class Path:
        def __truediv__(self,other):return self
        def exists(self):return terminal is not None
    monkeypatch.setattr(fixed.original,'ROOT',Path())
    monkeypatch.setattr(fixed.original,'read',lambda p:dict(state_root='external') if p=='post' else terminal)
    def forbidden(*args):raise AssertionError('Incomplete external review cannot be accepted')
    monkeypatch.setattr(fixed.recovery,'completed_review',forbidden)
    covered={};recovered={}
    fixed.adopt_external_recovery(dict(external_completed_recoveries=[dict(canonical_id='psm',post_review_configuration='post')]),
        {},dict(psm=({},{})),recovered,covered)
    assert covered==recovered=={}


def test_new_post_owner_is_identified_by_exact_native_job_without_dropping_old_live_ids(monkeypatch):
    command=['python','-m',fixed.POST_MODULE,'--config','post.json','--stage-worker']
    class Actor:
        info=dict(name='python.exe')
        def cmdline(self):return command
        def create_time(self):return 123.5
    monkeypatch.setattr(fixed.original,'live_stage_jobs',lambda:{'old-owner-job'})
    monkeypatch.setattr(fixed.original.psutil,'process_iter',lambda fields:[Actor()])
    monkeypatch.setattr(fixed.original,'read',lambda path:dict(phase='post',native_configuration='native.json')
        if path=='post.json' else dict(job_id='new-original-job'))
    assert fixed.live_stage_jobs()=={'old-owner-job','new-original-job'}
