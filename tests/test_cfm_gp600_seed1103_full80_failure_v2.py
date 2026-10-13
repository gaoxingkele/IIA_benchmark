import ast
import inspect
from scripts.flow_matching import audit_cfm_gp600_seed1103_full80_failure_v2 as current


def test_original_failure_verifier_changes_only_actual14trajectory_progress():
    tree=current.verify_ast()
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value==14:node.value=67
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(current.previous.verify)))
    assert current.verify.__globals__['IDENTITY']==current.IDENTITY


def test_same_original_full80_worker_and_observer_code():
    values=current.namespace();prior=current.previous.namespace()
    assert values['worker'].__code__.co_code==prior['worker'].__code__.co_code
    assert values['worker'].__globals__ is values and values['verify'] is current.verify
    assert values['RHSObserver'] is current.previous.previous.RHSObserver


def test_actual_saved_source_failure_and_unchanged_original_budget():
    config=current.read(current.ROOT/'configs/runtime/fm_cfm_GP6001103_full80_failure_observation.v2.json')
    job,model=current.verify(config)
    assert job['id']==current.IDENTITY and job['expected_optimizer_updates']==375000
    assert job['epochs']==600 and job['test_trajectories']==80
    assert config['settings']['minimum_commit_headroom_bytes']==30*2**30
