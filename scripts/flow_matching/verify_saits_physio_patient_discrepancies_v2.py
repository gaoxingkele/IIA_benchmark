"""Independently audit every full original patient array, retaining mismatches."""
import ast

from scripts.flow_matching import verify_saits_physio_completed_full_replay_v1 as legacy
from scripts.flow_matching.audit_saits_physio_patient_discrepancies_v2 import patient_statistics_review


def check_terminal(replay, receipt, producer):
    wanted = [r['id'] for r in producer['completed_jobs']]
    actual = [r['id'] for r in replay['records']]
    reviews = replay['original_patient_statistics_reviews']
    all_matched = all(r['passed'] for r in reviews)
    if (len(wanted) != producer['full_model_count'] or len(wanted) != len(set(wanted))
            or len(actual) != len(set(actual)) or set(actual) != set(wanted)
            or replay['full_models_replayed'] != producer['full_model_count']
            or len(actual) != producer['full_model_count']
            or len(reviews) != len(actual) or [r['id'] for r in reviews] != actual
            or not replay['all_registered_native_models_reinferred']
            or not replay['deferred_numerical_discrepancies_never_certify_reproduction']
            or replay['full_checkpoint_contents_and_re_inference_passed'] != all_matched
            or [r['original_patient_statistics_match'] for r in replay['records']] != [r['passed'] for r in reviews]
            or replay['cuda_context_initialized'] or replay['additional_training_seed_slots'] != 0
            or receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
            or receipt['full_native_model_replay_passed'] != all_matched):
        raise ValueError('Require every registered full native model and coordinate; numerical failure cannot be relabeled success')


def review_ast():
    tree = legacy.review_ast()
    original_check = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call)
            and child.exc.args and isinstance(child.exc.args[0], ast.Constant)
            and child.exc.args[0].value == 'Every original grouped error statistic must agree'
            for child in node.body))
    class Diagnostics(ast.NodeTransformer):
        checks = 0
        reports = 0
        rows = 0
        dispatches = 0
        producers = 0
        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.name == 'verify':
                node.body.insert(0, ast.parse('patient_reviews = []').body[0])
            return node
        def visit_If(self, node):
            if node is original_check:
                self.checks += 1
                return ast.parse('patient_reviews.append(patient_statistics_review(np.unique(groups), ids, grouped, original, producer["metric_tolerance"], native, None))').body[0]
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                keys = {k.arg for k in node.keywords}
                if 'full_native_models_verified' in keys:
                    self.reports += 1
                    for k in node.keywords:
                        if k.arg == 'passed': k.value = ast.parse('all(r["passed"] for r in patient_reviews)', mode='eval').body
                        if k.arg == 'full_native_models_verified': k.arg = 'full_native_models_coordinate_audited'
                    node.keywords.append(ast.keyword(arg='independent_original_patient_statistics_reviews',
                        value=ast.Name(id='patient_reviews', ctx=ast.Load())))
                if 'full_group_statistics_independently_recomputed' in keys:
                    self.rows += 1
                    node.keywords.append(ast.keyword(arg='original_patient_statistics_match',
                        value=ast.parse('patient_reviews[-1]["passed"]', mode='eval').body))
                    for k in node.keywords:
                        if k.arg == 'months': k.arg = 'patient_count'
            return node
        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.verify_saits_physio_completed_full_replay_v1':
                node.value = 'scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2'; self.dispatches += 1
            elif node.value == 'audit_saits_physio_full_checkpoint_replay_v1':
                node.value = 'audit_saits_physio_patient_discrepancies_v2'; self.producers += 1
            return node
    adapter = Diagnostics(); tree = adapter.visit(tree)
    if (adapter.checks, adapter.reports, adapter.rows, adapter.dispatches, adapter.producers) != (1, 1, 1, 1, 1):
        raise ValueError('Only patient diagnostics, report metadata and isolated dispatch may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(__name__='_independent_physio_patient_discrepancies',
        __file__=legacy.original.__file__, patient_statistics_review=patient_statistics_review)
    exec(compile(review_ast(), legacy.original.__file__, 'exec'), namespace)
    namespace['verify_record_identity'] = legacy.verify_physio_identity
    namespace['check_terminal'] = check_terminal
    namespace['main']()


if __name__ == '__main__':
    main()
