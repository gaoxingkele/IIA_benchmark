import ast
import inspect
from scripts.flow_matching import audit_cfm_main_BB6001106_full20_failure_v1 as current


def test_failure_binding_changes_only_actual_bb_identity_budget_and_progress():
    tree=current.verify_ast();restore={9600:93600,'bb':'gp',11:12}
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value in restore:node.value=restore[node.value]
    assert ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(current.previous.verify)))
    assert current.verify.__globals__['IDENTITY']==current.IDENTITY


def test_same_full20_worker_observer_controller_and_all_numeric_code():
    values=current.namespace();prior=current.previous.namespace()
    assert values['worker'].__code__.co_code==prior['worker'].__code__.co_code
    assert values['main'].__code__.co_code==prior['main'].__code__.co_code
    assert values['verify'] is current.verify and values['worker'].__globals__ is values
    assert values['RHSObserver'] is prior['RHSObserver']


def test_original_saved_bb_budget_and_nonresource_scoring_failure():
    cfg=current.read(current.ROOT/'configs/runtime/fm_cfm_main_BB6001106_full20_failure_observation.v1.json')
    job,model=current.verify(cfg)
    assert job['method']=='bb' and job['epochs']==600 and job['expected_optimizer_updates']==9600
    assert job['test_trajectories']==20 and job['test_points_per_trajectory']==500
    assert cfg['settings']['minimum_commit_headroom_bytes']==30*2**30
