"""Exact full stage recovery dispatch and immutable complete Sine replay route."""
import ast
import copy
import inspect

from scripts.flow_matching import run_timegan_corrected_stage_recovery_v3 as controller
from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_resume_compiler
from scripts.flow_matching import run_timegan_full45_completion_replay_v2 as replay


def config():
    return controller.original.read(controller.original.ROOT /
        'configs/reproducibility/fm_timegan_sine_stage_recovery.2026-10-12_v3.json')


def test_only_worker_module_dispatch_changes_in_original_guarded_recovery_controller():
    expected = ast.parse(inspect.getsource(controller.original.main))
    actual = controller.main_ast()
    changes = [0]

    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_corrected_stage_recovery_v3'" in node.value:
                node.value = node.value.replace('scripts.flow_matching.run_timegan_corrected_stage_recovery_v3',
                    'scripts.flow_matching.run_timegan_stage_resume_v2')
                changes[0] += 1
            return node

    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)
    assert changes == [1]


def test_exact_native_restore_proof_is_bound_before_full_training_recovery():
    value = config()
    function, queue, job = corrected_resume_compiler()(value)
    assert callable(function) and len(queue['jobs']) == 45
    assert job == dict(value['canonical_job'], output_directory=value['output_directory'])
    assert function.__globals__['_prior'] == dict(E0=50000, GS=50000, E=0, G=0, D=0)
    assert function.__globals__['_provenance']['discarded_unsaved_joint_updates_not_counted'] is True


def test_full45_route_now_includes_same_canonical_Sine_training_recovery_and_all_bitwise_gates():
    root = controller.original.ROOT
    value = controller.original.read(root / 'configs/runtime/fm_timegan_full45_completion_replay.v3.json')
    queue, routes = replay.runtime_namespace()['verify'](value)
    assert len(routes) == 45 and value['required_evaluators_each'] == 20
    assert value['required_checkpoints_each'] == 23 and value['equality_gate'] == 'bitwise'
    assert value['protocol_iterations_each'] == 70000
    selected = next(r for r in routes if r['canonical_job']['id'] == config()['canonical_job_id'])
    assert selected['kind'] == 'stage_resume'
    assert selected['execution_job']['output_directory'] == config()['output_directory']
    assert selected['native_completion_receipts'][0]['successful_field'] == 'full_native_recovery_result_verified'
    assert value['settings'] == queue['settings']


def test_other44_complete_model_routes_remain_identical_to_predecessor():
    root = controller.original.ROOT
    before = controller.original.read(root / 'configs/runtime/fm_timegan_full45_completion_replay.v2.json')
    after = controller.original.read(root / 'configs/runtime/fm_timegan_full45_completion_replay.v3.json')
    identity = config()['canonical_job_id']
    assert [r for r in before['routes'] if r['canonical_job']['id'] != identity] == [
        r for r in after['routes'] if r['canonical_job']['id'] != identity]
