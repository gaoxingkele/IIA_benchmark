"""Replay all ten original models and preserve strict patient-statistic failures.

Numerical discrepancies are recorded before proceeding to the next full model.
Identity, checkpoint, global metric, resource and full-coordinate checks remain
fatal. A numerical patient discrepancy can never produce a passing receipt.
"""
import ast
from pathlib import Path

from scripts.flow_matching import audit_saits_physio_full_checkpoint_replay_v1 as legacy


def patient_statistics_review(group_ids, unit_ids, grouped, saved, tolerance, native, target):
    import numpy as np
    from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import write
    expected = saved['statistics']
    if (not np.array_equal(group_ids, saved['group_ids'])
            or sorted(unit_ids.tolist()) != sorted(saved['unit_ids'].tolist())
            or grouped.shape != expected.shape or grouped.ndim != 2 or grouped.shape[1] != 4
            or not np.isfinite(grouped).all() or not np.isfinite(expected).all()):
        raise ValueError('All original patient identities and finite statistic shapes must agree')
    differences = np.abs(grouped - expected)
    allowed = tolerance['atol'] + tolerance['rtol'] * np.abs(expected)
    failed = ~np.isclose(grouped, expected, **tolerance)
    coordinates = np.argwhere(failed)
    report = dict(id=native['id'], seed=native['seed'], passed=bool(not failed.any()),
        original_patient_count=len(group_ids), unchanged_metric_tolerance=tolerance,
        columns=['absolute_error_sum', 'squared_error_sum', 'held_out_count', 'absolute_truth_sum'],
        failed_statistic_cells=int(failed.sum()), failed_patients=int(failed.any(axis=1).sum()),
        maximum_absolute_difference_per_column=differences.max(axis=0).tolist(),
        maximum_tolerance_ratio_per_column=(differences / allowed).max(axis=0).tolist(),
        all_original_patient_identities_equal=True,
        failures=[dict(patient_id=str(group_ids[row]), column=int(column),
            replay=float(grouped[row, column]), original=float(expected[row, column]),
            absolute_difference=float(differences[row, column]), allowed_difference=float(allowed[row, column]))
            for row, column in coordinates],
        additional_training_seed_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False,
        boundary='Every patient statistic uses the unchanged original tolerance. Differences are evidence, not accepted reproduction. No cause is inferred from aggregate metric agreement.')
    if target is not None:
        write(target / (native['id'] + '_patient_discrepancies.json'), report)
    return report


def replay_ast():
    tree = legacy.replay_ast()
    group_check = next(node for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call) and child.exc.args
            and isinstance(child.exc.args[0], ast.Constant)
            and child.exc.args[0].value == 'All original grouped error/target statistics must agree'
            for child in node.body))
    class Diagnostics(ast.NodeTransformer):
        dispatches = 0
        checks = 0
        reports = 0
        rows = 0
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name == 'worker':
                node.body.insert(0, ast.parse('patient_reviews = []').body[0])
            return node
        def visit_If(self, node):
            if node is group_check:
                self.checks += 1
                return ast.parse('patient_reviews.append(patient_statistics_review(group_ids, unit_ids, grouped, saved, config["metric_tolerance"], native, target))').body[0]
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                keys = {k.arg for k in node.keywords}
                if 'full_checkpoint_contents_and_re_inference_passed' in keys:
                    self.reports += 1
                    for keyword in node.keywords:
                        if keyword.arg == 'full_checkpoint_contents_and_re_inference_passed':
                            keyword.value = ast.parse('all(r["passed"] for r in patient_reviews)', mode='eval').body
                        elif keyword.arg == 'boundary':
                            keyword.value = ast.Constant(value='All registered complete original checkpoints reinferred on every original test coordinate with unchanged model/loader/global metric/resource checks. Strict patient-statistic differences are preserved across every model; any mismatch makes reproduction fail at the original tolerance. No cause, bitwise prediction equivalence, original BRITS2018, author-paper or TAB equivalence is inferred.')
                    node.keywords.extend([
                        ast.keyword(arg='all_registered_native_models_reinferred', value=ast.Constant(value=True)),
                        ast.keyword(arg='original_patient_statistics_reviews', value=ast.Name(id='patient_reviews', ctx=ast.Load())),
                        ast.keyword(arg='deferred_numerical_discrepancies_never_certify_reproduction', value=ast.Constant(value=True))])
                if 'full_test_grid_and_grouped_statistics_replayed' in keys:
                    self.rows += 1
                    node.keywords.append(ast.keyword(arg='original_patient_statistics_match',
                        value=ast.parse('patient_reviews[-1]["passed"]', mode='eval').body))
            return node
        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.audit_saits_physio_full_checkpoint_replay_v1':
                self.dispatches += 1
                node.value = 'scripts.flow_matching.audit_saits_physio_patient_discrepancies_v2'
            return node
    adapter = Diagnostics()
    tree = adapter.visit(tree)
    if (adapter.checks, adapter.reports, adapter.rows, adapter.dispatches) != (1, 1, 1, 1):
        raise ValueError('Only the registered strict patient diagnostic, report fields and worker dispatch may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(__name__='_saits_physio_patient_forensic_replay',
        __file__=legacy.original.__file__, patient_statistics_review=patient_statistics_review)
    exec(compile(replay_ast(), legacy.original.__file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
