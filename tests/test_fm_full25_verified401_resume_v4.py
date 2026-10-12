import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_full_image_training_execution_recovery_v4 as worker
from scripts.flow_matching import run_full_image_training_recovery_queue_v4 as queue


def config():return queue.original.read(queue.original.ROOT/'configs/reproducibility/fm_full_image_training_verified401_resume.2026-10-12_v4.json')


def test_remove_resume_observation_restores_original_full_training_AST_for_both_objectives():
    for filename in ('run_flow_mismatching_full_training_v1.py','run_wt_full_image_training_v1.py'):
        path=worker.original.ROOT/'scripts/flow_matching'/filename;tree=worker.training_ast(path)
        class Remove(ast.NodeTransformer):
            count=0
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_verify_full_resume':
                    self.count+=1;return None
                return self.generic_visit(node)
        remove=Remove();tree=remove.visit(tree)
        assert remove.count==1 and ast.dump(tree)==ast.dump(worker.original.training_ast(path))


def test_complete20FM5WT_scope_all2000epoch_budgets_original_lanes_and_archives_retained():
    value=config();registered=queue.verify_registration(value)
    assert len(registered)==25 and list(registered.values()).count('FM')==20 and list(registered.values()).count('WT')==5
    assert value['saved401_resume']['completed_epochs']==401 and value['saved401_resume']['remaining_epochs']==1599
    assert value['resource_lock']=='experiments/runs/fm_grasp_protocol_gpu.lock'


@pytest.mark.parametrize('mutation',['scope','epochs','guard','lane'])
def test_full_scope_budget_and_original_resource_safety_cannot_shrink(mutation):
    value=copy.deepcopy(config())
    if mutation=='scope':value['ordered_original_jobs'].pop()
    elif mutation=='epochs':value['saved401_resume']['remaining_epochs']-=1
    elif mutation=='guard':value['settings']['minimum_commit_headroom_bytes']-=1
    else:value['resource_lock']='other_GPU.lock'
    with pytest.raises(ValueError):queue.verify_registration(value)


def test_full_queue_controller_only_changes_next_fresh_worker_entry():
    tree=queue.main_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_full_image_training_execution_recovery_v4':
            node.value='scripts.flow_matching.run_full_image_training_execution_recovery_v3'
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(queue.original.main)))
