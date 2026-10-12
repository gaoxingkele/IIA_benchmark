import ast
import inspect

from scripts.flow_matching import run_full_image_training_execution_recovery_v4 as worker


def test_only_full_resume_observer_symbol_added_to_original_isolated_worker_namespace():
    tree=worker.run_ast();removed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            before=len(node.keywords)
            node.keywords=[item for item in node.keywords if item.arg!='_verify_full_resume']
            removed+=before-len(node.keywords)
    assert removed==1
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(worker.original.run)))


def test_restore_observation_has_no_optimizer_step_or_new_training_forward():
    tree=ast.parse(inspect.getsource(worker.verify_native_resume))
    names=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert 'step' not in names and 'backward' not in names
    assert 'get_rng_state_all' in names and 'get_rng_state' in names and 'state_dict' in names
