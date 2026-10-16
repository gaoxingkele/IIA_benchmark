import ast
import copy
from pathlib import Path
import pytest
from scripts.flow_matching import continue_spectral70_fair_conditions_v1 as s

def test_all_original_conditions_profiles_and_worker_commands_retained():
    c=s.registration();original,parent,cases=s.verify(c)
    assert len(cases)==c['required_conditions']==70 and c['required_cohorts']==14
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']
    assert c['spectral_flow_minimum_commit_headroom_bytes']==48*2**30
    assert c['inter_condition_yield_seconds']==35
    changed=copy.deepcopy(c);changed['required_conditions']=5
    with pytest.raises(ValueError,match='All70'):s.verify(changed)

def test_handoff_rejects_live_children_unknown_partial_attempts_and_identity_drift():
    h=s.EXPECTED;state=dict(pid=h['pid'],create_time=h['create_time'],required_conditions=70,required_cohorts=14,
        state='scanning_all70_full_source_proofs',verified_ids=['done'],failed_preserved={'bad':'preserved'})
    assert s.boundary_allowed(state,h,[],True,['done','bad'])
    assert not s.boundary_allowed(state,h,[],True,['partial'])
    assert not s.boundary_allowed(state,h,[],False,['done'])
    assert not s.boundary_allowed(state,h,[dict(name='python.exe',exe='python.exe')],True,['done'])
    for key,value in [('pid',1),('create_time',0),('required_conditions',5),('required_cohorts',1)]:
        assert not s.boundary_allowed(dict(state,**{key:value}),h,[],True,['done'])

def test_only_metadata_directory_lock_restore_and_yield_nodes_change_original_program():
    original=ast.parse(Path(s.original.__file__).read_text(encoding='utf-8'))
    before=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    _,after=s.controller_program(dict(state_root='unchanged'),dict(verified_ids=[],failed_preserved={}),{},None,Path('metadata'))
    def numerical_calls(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and (
            isinstance(n.func,ast.Attribute) and n.func.attr in ['Popen','exclusive'] or
            isinstance(n.func,ast.Subscript) and isinstance(n.func.slice,ast.Constant) and n.func.slice.value=='guarded_gpu_wait')]
    assert numerical_calls(before)==numerical_calls(after)
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_yield_completed' for n in ast.walk(after))

def test_restored_state_keeps_all_successes_and_failures_and_never_changes_zero_training_credit(monkeypatch):
    monkeypatch.setattr(s,'sha',lambda path:'fixed')
    original=dict(pid=99,required_conditions=70,required_cohorts=14,new_generator_optimizer_updates=0,new_training_seed_slots=0)
    old=dict(verified_ids=['a','b'],failed_preserved={'failed':'keep'})
    value=s.initial_state(original,old)
    assert value['verified_ids']==['a','b'] and value['failed_preserved']==old['failed_preserved']
    assert value['new_generator_optimizer_updates']==value['new_training_seed_slots']==0
    assert old==dict(verified_ids=['a','b'],failed_preserved={'failed':'keep'})
