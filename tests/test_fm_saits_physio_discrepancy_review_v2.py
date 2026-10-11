import ast
from copy import deepcopy

import pytest

from scripts.flow_matching import verify_saits_physio_completed_full_replay_v1 as legacy
from scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2 import check_terminal, review_ast


def documents():
    producer = dict(full_model_count=2, completed_jobs=[dict(id='a'), dict(id='b')])
    replay = dict(records=[dict(id='a', original_patient_statistics_match=True), dict(id='b', original_patient_statistics_match=False)],
        original_patient_statistics_reviews=[dict(id='a', passed=True), dict(id='b', passed=False)],
        full_models_replayed=2, all_registered_native_models_reinferred=True,
        deferred_numerical_discrepancies_never_certify_reproduction=True,
        full_checkpoint_contents_and_re_inference_passed=False,
        cuda_context_initialized=False, additional_training_seed_slots=0)
    receipt = dict(exit_code=0, resource_guard_aborted=False, full_native_model_replay_passed=False)
    return replay, receipt, producer


def test_audit_accepts_complete_failed_reproduction_for_analysis_without_relabeling():
    replay, receipt, producer = documents()
    check_terminal(replay, receipt, producer)
    assert not replay['full_checkpoint_contents_and_re_inference_passed']


@pytest.mark.parametrize('change', ['missing_model', 'duplicate_model', 'missing_review', 'swapped_review',
    'budget', 'aborted', 'worker_failure', 'false_certification', 'false_receipt', 'incomplete_inference', 'extra_seed', 'record_disagreement'])
def test_incomplete_or_misrepresented_replay_cannot_enter_full_array_audit(change):
    replay, receipt, producer = documents()
    if change == 'missing_model': replay['records'].pop()
    elif change == 'duplicate_model': replay['records'][1]['id'] = 'a'
    elif change == 'missing_review': replay['original_patient_statistics_reviews'].pop()
    elif change == 'swapped_review': replay['original_patient_statistics_reviews'].reverse()
    elif change == 'budget': producer['full_model_count'] = 3
    elif change == 'aborted': receipt['resource_guard_aborted'] = True
    elif change == 'worker_failure': receipt['exit_code'] = 1
    elif change == 'false_certification': replay['full_checkpoint_contents_and_re_inference_passed'] = True
    elif change == 'false_receipt': receipt['full_native_model_replay_passed'] = True
    elif change == 'incomplete_inference': replay['all_registered_native_models_reinferred'] = False
    elif change == 'extra_seed': replay['additional_training_seed_slots'] = 1
    elif change == 'record_disagreement': replay['records'][1]['original_patient_statistics_match'] = True
    with pytest.raises(ValueError): check_terminal(replay, receipt, producer)


def test_all_native_truth_mask_coordinate_metric_checkpoint_and_resource_logic_is_preserved():
    old = legacy.review_ast()
    original_check = next(node for node in ast.walk(old) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call)
            and child.exc.args and isinstance(child.exc.args[0], ast.Constant)
            and child.exc.args[0].value == 'Every original grouped error statistic must agree'
            for child in node.body))
    class Undo(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name == 'verify':
                assert ast.unparse(node.body[0]) == 'patient_reviews = []'
                node.body.pop(0)
            return node
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'patient_reviews':
                return deepcopy(original_check)
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                is_report = any(k.arg == 'full_native_models_coordinate_audited' for k in node.keywords)
                node.keywords = [k for k in node.keywords if k.arg not in {'independent_original_patient_statistics_reviews', 'original_patient_statistics_match'}]
                for k in node.keywords:
                    if k.arg == 'full_native_models_coordinate_audited': k.arg = 'full_native_models_verified'
                    if k.arg == 'patient_count': k.arg = 'months'
                    if is_report and k.arg == 'passed': k.value = ast.Constant(value=True)
            return node
        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2': node.value = 'scripts.flow_matching.verify_saits_physio_completed_full_replay_v1'
            if node.value == 'audit_saits_physio_patient_discrepancies_v2': node.value = 'audit_saits_physio_full_checkpoint_replay_v1'
            return node
    assert ast.dump(Undo().visit(review_ast())) == ast.dump(old)


def test_independent_report_keeps_any_numeric_patient_failure_failed():
    report = next(node for node in ast.walk(review_ast()) if isinstance(node, ast.Call)
        and any(k.arg == 'full_native_models_coordinate_audited' for k in node.keywords))
    value = next(k.value for k in report.keywords if k.arg == 'passed')
    expression = compile(ast.fix_missing_locations(ast.Expression(body=value)), '<report>', 'eval')
    assert not eval(expression, {'patient_reviews': [{'passed': True}, {'passed': False}]})
