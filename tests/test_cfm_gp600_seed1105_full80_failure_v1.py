import ast
import inspect
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as current


def test_observer_and_full_solver_worker_change_only_optimizer_budget_binding():
    tree=current.worker_ast()
    class Restore(ast.NodeTransformer):
        def visit_Subscript(self,node):
            if ast.dump(node)==ast.dump(ast.parse("job['expected_optimizer_updates']",mode='eval').body):
                return ast.Constant(value=37800)
            return self.generic_visit(node)
    tree=Restore().visit(tree)
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(current.previous.worker)))
    values=current.namespace()
    assert values['RHSObserver'] is current.previous.RHSObserver
    assert values['verify'] is current.verify


def test_actual_failed_receipt_and_original_full_gp_budget_are_required():
    config=current.read(current.ROOT/'configs/runtime/fm_cfm_GP6001105_full80_failure_observation.v1.json')
    job,model=current.verify(config)
    assert job['epochs']==600 and job['expected_optimizer_updates']==375000
    assert job['test_trajectories']==80 and job['test_points_per_trajectory']==500
    assert config['settings']['minimum_commit_headroom_bytes']==30*2**30
    assert config['solver_parameters']==dict(solver='dopri5',rtol=1e-7,atol=1e-9,max_num_steps=10000)
