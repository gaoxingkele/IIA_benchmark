import ast
from copy import deepcopy

import numpy as np
import pytest

from scripts.flow_matching import audit_saits_physio_full_checkpoint_replay_v1 as legacy
from scripts.flow_matching.audit_saits_physio_patient_discrepancies_v2 import patient_statistics_review, replay_ast


def case():
    groups = np.array(['p1', 'p2'])
    statistics = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    return groups, statistics, {'group_ids': groups.copy(), 'unit_ids': groups.copy(), 'statistics': statistics.copy()}


def check(tmp_path, groups, statistics, saved):
    return patient_statistics_review(groups, groups, statistics, saved,
        dict(rtol=2e-6, atol=2e-6), dict(id='original', seed=26), tmp_path)


def test_matching_all_patient_statistics_can_pass(tmp_path):
    groups, statistics, saved = case()
    report = check(tmp_path, groups, statistics, saved)
    assert report['passed'] and report['failed_statistic_cells'] == 0


def test_global_cancellation_never_accepts_patient_failures(tmp_path):
    groups, statistics, saved = case()
    statistics[0, 0] += .01
    statistics[1, 0] -= .01
    assert np.allclose(statistics.sum(axis=0), saved['statistics'].sum(axis=0))
    report = check(tmp_path, groups, statistics, saved)
    assert not report['passed']
    assert report['failed_statistic_cells'] == report['failed_patients'] == 2
    assert len(report['failures']) == 2
    assert report['maximum_tolerance_ratio_per_column'][0] > 1


@pytest.mark.parametrize('change', ['group_order', 'unit_identity', 'shape', 'nan', 'inf'])
def test_identity_or_invalid_statistics_remain_fatal(tmp_path, change):
    groups, statistics, saved = case()
    if change == 'group_order': saved['group_ids'] = groups[::-1]
    elif change == 'unit_identity': saved['unit_ids'] = np.array(['p1', 'other'])
    elif change == 'shape': saved['statistics'] = np.ones((2, 3))
    elif change == 'nan': statistics[0, 0] = np.nan
    elif change == 'inf': saved['statistics'][0, 1] = np.inf
    with pytest.raises(ValueError): check(tmp_path, groups, statistics, saved)


def test_complete_native_source_is_preserved_after_removing_only_diagnostic_bindings():
    old = legacy.replay_ast()
    original_check = next(node for node in ast.walk(old) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call)
            and child.exc.args and isinstance(child.exc.args[0], ast.Constant)
            and child.exc.args[0].value == 'All original grouped error/target statistics must agree'
            for child in node.body))
    original_report = next(node for node in ast.walk(old) if isinstance(node, ast.Call)
        and any(k.arg == 'full_checkpoint_contents_and_re_inference_passed' for k in node.keywords))
    original_boundary = next(k.value for k in original_report.keywords if k.arg == 'boundary')
    class Undo(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name == 'worker':
                assert ast.unparse(node.body[0]) == 'patient_reviews = []'
                node.body.pop(0)
            return node
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'patient_reviews'):
                return deepcopy(original_check)
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                is_report = any(k.arg == 'full_checkpoint_contents_and_re_inference_passed' for k in node.keywords)
                node.keywords = [k for k in node.keywords if k.arg not in {
                    'all_registered_native_models_reinferred', 'original_patient_statistics_reviews',
                    'deferred_numerical_discrepancies_never_certify_reproduction', 'original_patient_statistics_match'}]
                for k in node.keywords:
                    if k.arg == 'full_checkpoint_contents_and_re_inference_passed': k.value = ast.Constant(value=True)
                    if is_report and k.arg == 'boundary': k.value = deepcopy(original_boundary)
            return node
        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.audit_saits_physio_patient_discrepancies_v2':
                node.value = 'scripts.flow_matching.audit_saits_physio_full_checkpoint_replay_v1'
            return node
    assert ast.dump(Undo().visit(replay_ast())) == ast.dump(old)


def test_failed_patient_review_cannot_create_passing_full_report():
    report = next(node for node in ast.walk(replay_ast()) if isinstance(node, ast.Call)
        and any(k.arg == 'full_checkpoint_contents_and_re_inference_passed' for k in node.keywords))
    value = next(k.value for k in report.keywords if k.arg == 'full_checkpoint_contents_and_re_inference_passed')
    expression = compile(ast.fix_missing_locations(ast.Expression(body=value)), '<report>', 'eval')
    assert eval(expression, {'patient_reviews': [{'passed': True}]})
    assert not eval(expression, {'patient_reviews': [{'passed': True}, {'passed': False}]})
