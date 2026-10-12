import ast

import pytest

from scripts.flow_matching import run_timegan_full45_completion_replay_v3 as current
from scripts.flow_matching import run_timegan_full45_completion_replay_v2 as previous


def test_all_native_full_replay_workers_and_bitwise_gates_keep_the_original_code():
    namespace = current.runtime_namespace()
    for name in ('worker', 'check_report', 'complete_metadata', 'restore_full_checkpoint',
                 'successful_native_receipt', 'bitwise_array', 'bitwise_score', 'independent_full_loader'):
        assert namespace[name].__code__ is previous.runtime_namespace()[name].__code__


def test_full45_controller_keeps_all_source_logic_and_original_guards():
    original = previous.main_ast()
    observed = current.main_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str):
                n.value = n.value.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v3',
                                          'scripts.flow_matching.run_timegan_full45_completion_replay_v2')
            return n
        def visit_Assign(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == '_cached_resource_api':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api', asname=None)], level=0)
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(observed)) == ast.dump(original)


def idle_state():
    return dict(state='waiting_for_memory', active_pid=None, active_job=None, verified_original_ids=[],
                failed={}, required_original_models=45, pending_original_models=45)


def test_actual_idle_capacity_wait_is_safe_for_controller_only_handoff():
    assert current.handoff_phase_allowed('original', idle_state(), False,
        [dict(name='conhost.exe', exe='C:/Windows/System32/conhost.exe')])


@pytest.mark.parametrize('corruption', ['child','active_pid','active_job','verified','failed','scope','training_phase'])
def test_handoff_rejects_native_work_or_completion_history(corruption):
    state = idle_state()
    descendants = []
    if corruption == 'child': descendants.append(dict(name='python.exe', exe='F:/native/python.exe'))
    if corruption == 'active_pid': state['active_pid'] = 1
    if corruption == 'active_job': state['active_job'] = 'source_job'
    if corruption == 'verified': state['verified_original_ids'] = ['source_job']
    if corruption == 'failed': state['failed'] = {'source_job':'bitwise mismatch'}
    if corruption == 'scope': state['pending_original_models'] = 44
    if corruption == 'training_phase': state['state'] = 'running_complete_full45_native_case_replay'
    assert not current.handoff_phase_allowed('original', state, False, descendants)
