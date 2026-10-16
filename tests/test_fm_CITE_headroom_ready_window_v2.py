import ast
import inspect
from pathlib import Path
import pytest
from scripts.flow_matching import reserve_CITE_headroom_ready_window_v2 as s


def api(headroom):
    return dict(resource_snapshot=lambda:dict(physical_available_bytes=20,commit_available_bytes=40),
        has_headroom=lambda snapshot,settings:headroom)


def test_insufficient_actual_headroom_never_requests_either_family_lock():
    called=[]
    assert s.headroom_gated_reserve(lambda path:called.append(path),[Path('a'),Path('b')],api(False),{}) is None
    assert called==[]


def test_original_physical_and_commit_requirements_are_both_enforced():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    gate=resource_api()['has_headroom'];settings=s.original.registration()['original_preparation_settings']
    called=[];a=dict(resource_snapshot=lambda:dict(physical_available_bytes=settings['minimum_available_memory_bytes'],
        commit_available_bytes=settings['minimum_commit_headroom_bytes']-1),has_headroom=gate)
    assert s.headroom_gated_reserve(lambda path:called.append(path),[Path('a'),Path('b')],a,settings) is None
    assert called==[]


def test_actual_ready_headroom_retains_atomic_two_lock_rollback():
    class Handle:
        closed=False
        def close(self):self.closed=True
    h=Handle();calls=[]
    def lock(path):calls.append(path);return h if len(calls)==1 else None
    assert s.headroom_gated_reserve(lock,[Path('b'),Path('a')],api(True),{}) is None
    assert calls==[Path('a'),Path('b')] and h.closed


def test_main_is_original_finite_window_except_one_gated_reservation_and_status_text():
    baseline=ast.parse(inspect.getsource(s.original.main));changed=s.main_tree()
    class Reverse(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='headroom_gated_reserve':n.func.id='try_reserve';n.args=n.args[:2]
            return n
        def visit_Constant(self,n):
            if n.value=='waiting_original_headroom_and_both_families_idle_no_lock_owned':n.value='waiting_both_families_naturally_idle_no_lock_owned'
            return n
    assert ast.dump(baseline)==ast.dump(Reverse().visit(changed))


def test_all_original_deadlines_guards_and_same_preparation_owner_preserved():
    old=s.original.registration();new=s.registration()
    for key in ['preparation_configuration','preparation_handle','family_locks','maximum_lock_acquisition_seconds',
        'maximum_owned_window_seconds','initial_headroom_grace_seconds','original_preparation_settings',
        'original_preparation_stable_samples','original_preparation_poll_seconds']:
        assert old[key]==new[key]
    assert new['require_original_headroom_before_reserving_any_family_lock']
    assert not new['numerical_preprocessing_attempt_repeated']
    assert new['original_preparation_settings']['minimum_commit_headroom_bytes']==32*2**30
    assert new['original_preparation_stable_samples']==6 and new['original_preparation_poll_seconds']==5
