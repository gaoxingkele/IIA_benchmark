import ast
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching.audit_rcgan_random_order_chain_v3 import full366_chain_ast
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4 import full366_actual358_trace_ast
from scripts.flow_matching import rcgan_segmented_checkpoint_evidence_v1 as evidence
from scripts.flow_matching.rcgan_seed1104_resume_binding_v3 import full_resume_ast


def test_all_original_random_draw_and_guard_operations_are_preserved():
    source = Path('scripts/flow_matching/audit_rcgan_random_order_chain_v1.py').read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name in ('worker', 'main')]
    actual = full366_chain_ast(source)
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, n):
            if type(n.value) is int and n.value == 366:
                n.value = 238
            elif isinstance(n.value, str):
                n.value = n.value.replace('366', '238').replace(
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v3',
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v1')
            return n
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_all_original_checkpoint_math_guards_and_bitwise_gates_are_preserved():
    source = Path('scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py').read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('verify_registration', 'worker', 'main')]
    actual = full366_actual358_trace_ast(source)
    class Reverse(ast.NodeTransformer):
        def visit_Call(self, n):
            if isinstance(n.func, ast.Name) and n.func.id == 'actual_trace_epochs':
                return ast.parse('list(range(231))', mode='eval').body
            return self.generic_visit(n)
        def visit_Constant(self, n):
            if type(n.value) is int:
                n.value = {366: 238, 358: 231}.get(n.value, n.value)
            elif isinstance(n.value, str):
                n.value = n.value.replace('epoch_365', 'epoch_237').replace('366', '238').replace('358', '231').replace(
                    '0..80 + 84..360; eight missing rows retained', '0..230').replace('eight', 'seven').replace(
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4',
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1')
            return n
        def visit_keyword(self, n):
            if n.arg:
                n.arg = n.arg.replace('all366_', 'all238_').replace('all358_', 'all231_').replace('eight_missing', 'seven_missing')
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


@pytest.mark.parametrize('builder', [full366_chain_ast, full366_actual358_trace_ast])
def test_partial_audit_implementation_is_rejected(builder):
    with pytest.raises(ValueError):
        builder('def main(): pass')


def test_exact_real_trace_gap_is_preserved():
    assert len(evidence.TRACE_EPOCHS) == 358
    assert set(range(366)) - set(evidence.TRACE_EPOCHS) == set(evidence.MISSING_TRACE_EPOCHS)
    assert evidence.MISSING_TRACE_EPOCHS == (81, 82, 83, 361, 362, 363, 364, 365)


@pytest.fixture
def manifest_fixture(monkeypatch):
    evaluations = [dict(epoch=e, sigma_optimizer_updates=2, complete_generated_points=4984,
        full_fit_points=996, full_test_points=3988, stopping_delta=0.0) for e in range(366)]
    restoration = dict(every_model_optimizer_variable_bitwise_restored=True,
        full36000_training_image_order_and_both_random_states_restored=True,
        completed_original_epochs=84, next_original_epoch=84, remaining_full_epochs=416,
        proof_binding=dict(checkpoint_state_proof_sha256='bound'))
    data = dict(chain=dict(source_receipts=[]), state=dict(source_receipts=[]),
        progress=dict(full_epochs_completed=366, required_epochs=500, full_training_complete=False,
            job=dict(seed=1104), optimizer_updates=dict(D=469944, G=469944, sigma=732)),
        terminal=dict(exit_code=1, full_original500epoch_resume_verified=False, resource_guard_aborted=True),
        evaluations=evaluations, epochs=[dict(epoch=e) for e in range(366)],
        prior=dict(passed=True, epochs=[dict(epoch=e) for e in range(84)],
            all84_full_model_and_optimizer_checkpoints_restored_bitwise=True,
            all81_persisted_native_trace_losses_reproduced_bitwise_float32=True),
        restoration=restoration, first_evaluations=deepcopy(evaluations[:84]))
    manifest = dict(completed_original_epochs=366, remaining_original_epochs=134, required_original_epochs=500,
        persisted_trace_epochs=list(evidence.TRACE_EPOCHS), missing_trace_epochs=list(evidence.MISSING_TRACE_EPOCHS),
        benchmark_performance=False, additional_training_seed_slots=0, source_receipts=[], derived_receipts=[],
        prior_chain_configuration='chain', prior_state_configuration='state', interrupted_progress='progress',
        failed_terminal_receipt='terminal', interrupted_job=dict(seed=1104), completed_evaluators='evaluations',
        completed_epochs='epochs', prior_checkpoint_proof='prior', first_resume_restoration='restoration',
        first_completed_evaluators='first_evaluations')
    monkeypatch.setattr(evidence, 'read', lambda p: data[Path(p).name])
    monkeypatch.setattr(evidence, 'sha', lambda p: 'bound')
    return manifest, data


def test_complete_segment_binding_is_accepted(manifest_fixture):
    manifest, data = manifest_fixture
    assert evidence.verify_manifest(manifest) == data['evaluations']


@pytest.mark.parametrize('corruption', ['short_budget', 'fabricated_time', 'guard_not_terminal',
                                     'missing_evaluator', 'counter_mismatch', 'failed_restore', 'prefix_changed'])
def test_weaker_recovery_evidence_is_rejected(manifest_fixture, corruption):
    manifest, data = manifest_fixture
    if corruption == 'short_budget': manifest['required_original_epochs'] = 366
    if corruption == 'fabricated_time': manifest['persisted_trace_epochs'].append(365)
    if corruption == 'guard_not_terminal': data['terminal']['exit_code'] = 0
    if corruption == 'missing_evaluator': data['evaluations'].pop()
    if corruption == 'counter_mismatch': data['progress']['optimizer_updates']['D'] -= 1
    if corruption == 'failed_restore': data['restoration']['every_model_optimizer_variable_bitwise_restored'] = False
    if corruption == 'prefix_changed': data['first_evaluations'][83]['sigma_optimizer_updates'] = 3
    with pytest.raises(ValueError): evidence.verify_manifest(manifest)


@pytest.mark.parametrize('path', ['rcgan_native_training_resume_runtime_v1.py', 'run_rcgan_native_state_recovery_v1.py'])
def test_full_resume_has_only_epoch_and_verifier_host_telemetry_bindings(path):
    source = (Path('scripts/flow_matching') / path).read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)]
    actual = full_resume_ast(source)
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, n):
            if type(n.value) is int:
                n.value = {366: 238, 134: 262}.get(n.value, n.value)
            elif isinstance(n.value, str):
                n.value = n.value.replace('366', '238').replace('134', '262').replace('358', '231').replace(
                    'epoch_365', 'epoch_237').replace('__state_recovery_v3', '__state_recovery_v1').replace(
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v3',
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v1')
            return n
        def visit_ImportFrom(self, n):
            if n.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v3':
                n.module = 'scripts.flow_matching.rcgan_native_training_resume_runtime_v1'
            return n
        def visit_Assign(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == 'state_verify_registration':
                alias = n.targets[0].id
                return ast.ImportFrom(module='scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1',
                    names=[ast.alias(name='verify_registration', asname=alias if alias != 'verify_registration' else None)], level=0)
            if isinstance(n.value, ast.Name) and n.value.id == '_cached_resource_api':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api', asname=None)], level=0)
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)
