import ast
import inspect

import pytest

from scripts.flow_matching import run_cfm_completion_replay_queue_v2 as current


def idle_state():
    return dict(active_pid=None,active_job=None,required_original_models=285,pending_original_models=200,
        verified_original_ids=[str(i) for i in range(85)],failed_preserved={},state='waiting_actual_original_training_completion')


def test_all_native_grid_model_and_review_functions_are_unchanged():
    values=current.namespace()
    for name in ('verify','verify_record','native_registration','review_worker','check_native_terminal','main'):
        assert values[name].__code__ is getattr(current.original,name).__code__
    assert values['main'].__globals__['controller'] is values['controller']


def test_full285_controller_has_only_idle_handoff_and_cached_telemetry():
    expected=ast.parse(inspect.getsource(current.original.controller))
    observed=current.controller_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Call(self,n):
            if getattr(n.func,'id',None)=='_cached_resource_api':n.func.id='resource_api'
            return self.generic_visit(n)
        def visit_UnaryOp(self,n):
            if ast.dump(n)==ast.dump(ast.parse('not _candidate_ready(route)',mode='eval').body):
                return ast.parse("not (ROOT/route['result_path']).is_file()",mode='eval').body
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and getattr(n.value.func,'id',None)=='_verified_handoff':return None
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(observed))==ast.dump(expected)


def test_idle_handoff_retains_the_existing85_independent_certificates():
    assert current.allowed_idle(idle_state(),[])


@pytest.mark.parametrize('corruption',['child','active_pid','active_job','new_verified_case','failed_case','scope_cut','running_phase'])
def test_handoff_rejects_any_live_numerical_work_or_changed_certificate_history(corruption):
    state=idle_state();children=[]
    if corruption=='child':children=[dict(name='python.exe',exe='F:/native/python.exe')]
    if corruption=='active_pid':state['active_pid']=1
    if corruption=='active_job':state['active_job']='source'
    if corruption=='new_verified_case':state['verified_original_ids'].append('extra')
    if corruption=='failed_case':state['failed_preserved']={'source':'bitwise mismatch'}
    if corruption=='scope_cut':state['required_original_models']=1
    if corruption=='running_phase':state['state']='running_full_original_checkpoint_replay'
    assert not current.allowed_idle(state,children)


def test_checkpointed_case_requires_actual_full_success_terminal(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(current.original,'ROOT',tmp_path)
    (tmp_path/'result.json').write_text('{}')
    route=dict(result_path='result.json',checkpointed_training_terminal='terminal.json')
    assert not current.candidate_ready(route)
    terminal=dict(exit_code=1,complete=False,full_original400_checkpointed_result_verified=False,
        resource_observations=dict(resource_guard_aborted=True))
    (tmp_path/'terminal.json').write_text(json.dumps(terminal))
    assert not current.candidate_ready(route)
    terminal.update(exit_code=0,complete=True,full_original400_checkpointed_result_verified=True)
    terminal['resource_observations']['resource_guard_aborted']=False
    (tmp_path/'terminal.json').write_text(json.dumps(terminal))
    assert current.candidate_ready(route)
