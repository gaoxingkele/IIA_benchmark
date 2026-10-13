import ast
import inspect
from scripts.flow_matching import audit_cfm_main_GP6001104_full20_failure_v1 as current


def restore_coverage_and_names(tree):
    class Restore(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==20:node.value=80
            if isinstance(node.value,str):node.value=node.value.replace('full20_','full80_')
            return node
        def visit_keyword(self,node):
            node.arg=node.arg.replace('full20_','full80_') if node.arg else node.arg
            return self.generic_visit(node)
    return Restore().visit(tree)


def test_full20_worker_changes_only_original_grid_capacity_and_artifact_name():
    assert ast.dump(restore_coverage_and_names(current.worker_ast()))==ast.dump(current.previous.worker_ast())
    values=current.namespace()
    assert values['verify'] is current.verify
    assert values['RHSObserver'] is current.previous.previous.RHSObserver
    assert values['worker'].__globals__ is values


def test_main_and_postaudit_preserve_all_original_checks_and_guards():
    assert ast.dump(restore_coverage_and_names(current.main_ast()))==ast.dump(ast.parse(inspect.getsource(current.previous.previous.main)))
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    assert ast.dump(restore_coverage_and_names(current.post_ast()))==ast.dump(arrays.corrected_audit_ast())


def test_actual_main_recipe_failure_and_original_full_budget_binding():
    config=current.read(current.ROOT/'configs/runtime/fm_cfm_main_GP6001104_full20_failure_observation.v1.json')
    job,model=current.verify(config)
    assert job['expected_optimizer_updates']==93600 and job['batch_size']==258
    assert job['test_trajectories']==20 and job['test_points_per_trajectory']==500
    assert config['settings']['minimum_commit_headroom_bytes']==30*2**30
