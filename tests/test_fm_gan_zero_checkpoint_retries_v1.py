"""Strict complete-budget zero-checkpoint retry registration and source fidelity."""
import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v4 as tg
from scripts.flow_matching import run_rcgan_zero_checkpoint_full_retry_v1 as rc
from scripts.flow_matching.gan_zero_checkpoint_retry_v1 import execution_job, stronger_admission, verify_zero_checkpoint_incident
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read


def configurations():
    return (read(ROOT / 'configs/runtime/fm_timegan_corrected_observer_dispatch.v4.json'),
            read(ROOT / 'configs/runtime/fm_rcgan_seed1107_zero_checkpoint_full_retry.v1.json'))


@pytest.mark.parametrize('index', [0, 1])
def test_actual_terminal_incidents_require_full_fresh_budget_without_credits(index):
    config = configurations()[index]
    receipt = verify_zero_checkpoint_incident(config['zero_checkpoint_retry'])
    assert receipt['resource_guard_aborted'] is True and receipt['exit_code'] == 1
    assert config['zero_checkpoint_retry']['completed_training_budget_credited'] == 0
    assert config['zero_checkpoint_retry']['failed_unsaved_optimizer_updates'] == 'not_recorded'
    assert config['zero_checkpoint_retry']['stopped_native_identity']['pid'] is None


def test_all45_timegan_jobs_keep_every_field_except_one_output():
    config, _ = configurations()
    queue = tg.verify_runtime(config)
    actual = tg.execution_jobs(config, queue['jobs'])
    assert len(actual) == 45
    changed = [i for i, (a,b) in enumerate(zip(actual,queue['jobs'])) if a != b]
    assert len(changed) == 1
    index = changed[0]
    expected = copy.deepcopy(queue['jobs'][index])
    expected['output_directory'] = config['zero_checkpoint_retry']['execution_output_directory']
    assert actual[index] == expected
    assert actual[index]['id'] == 'timegan_native__gru__sine__seed1104'


def test_rcgan_keeps_all_five_canonical_seeds_and_full500_budget():
    _, config = configurations()
    queue = rc.verify(config)
    assert len(queue['jobs']) == 5
    assert queue['epochs'] == queue['full_evaluations_per_job'] == 500
    assert sorted(job['seed'] for job in queue['jobs']) == list(range(1103,1108))
    assert queue['D_updates_per_epoch'] == queue['G_updates_per_epoch'] == 1284
    assert queue['sigma_maximum_updates_per_evaluation'] == 2000


@pytest.mark.parametrize('index', [0, 1])
def test_only_admission_increases_and_all_emergency_caps_stay(index):
    config = configurations()[index]
    settings = config['roles']['original']['settings'] if index == 0 else rc.verify(config)['settings']
    stronger = stronger_admission(config, settings)
    assert stronger['minimum_commit_headroom_bytes'] == 40 * 1024 ** 3
    assert stronger['emergency_commit_headroom_bytes'] == 16 * 1024 ** 3
    assert {k:v for k,v in settings.items() if k != 'minimum_commit_headroom_bytes'} == {
        k:v for k,v in stronger.items() if k != 'minimum_commit_headroom_bytes'}


def test_timegan_worker_scientific_statements_and_verifier_are_unchanged():
    from scripts.flow_matching.timegan_native_result_verifier_v2 import observer_worker_ast
    expected = observer_worker_ast()
    actual = tg.worker_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_execution_job':
                return self.visit(node.args[1])
            return self.generic_visit(node)
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_timegan_controller_changes_only_route_output_binding_and_admission():
    expected = tg.original.main_ast()
    actual = tg.main_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace('run_timegan_corrected_observer_dispatch_v4', 'run_timegan_corrected_observer_dispatch_v3')
            return node
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in ('_execution_jobs', '_stronger_admission'):
                return self.visit(node.args[1])
            return self.generic_visit(node)
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_rcgan_controller_preserves_every_original_scientific_operation():
    expected = ast.parse(inspect.getsource(rc.original.main))
    actual = rc.main_ast()
    class Reverse(ast.NodeTransformer):
        def visit_ListComp(self, node):
            return ast.parse("queue['jobs']", mode='eval').body
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_stronger_admission':
                return self.visit(node.args[1])
            return self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module == 'scripts.flow_matching.rcgan_seed1105_resume_binding_v1':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api')], level=0)
            return node
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_original_checkpoint_guard_cannot_be_skipped_by_host():
    config, _ = configurations()
    with pytest.raises(ValueError):
        verify_zero_checkpoint_incident(config['zero_checkpoint_retry'], check_live=False)


def test_full_replay_and_paper_equivalence_remain_unclaimed():
    evidence = read(ROOT / 'projects/flow_matching_research/results/2026-10-13/gan_zero_checkpoint_commit_retries_v1/registered_complete_retry_obligations.json')
    assert evidence['full_independent_replay_routes_require_completion_binding_for_these_new_outputs'] is True
    assert evidence['actual_full_training_completed'] is False
    assert evidence['additional_training_seed_slots'] == 0
    assert evidence['all_original_experiments_complete'] is False
