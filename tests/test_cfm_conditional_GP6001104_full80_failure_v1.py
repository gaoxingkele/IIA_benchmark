import ast
import inspect
from scripts.flow_matching import audit_cfm_conditional_GP6001104_full80_failure_v1 as current


def test_only_actual_failed_progress_predicate_changes():
    original=ast.parse(inspect.getsource(current.previous.verify));adapted=current.verify_ast()
    before=[ast.unparse(n) for n in ast.walk(original) if isinstance(n,ast.Compare)]
    after=[ast.unparse(n) for n in ast.walk(adapted) if isinstance(n,ast.Compare)]
    assert len(before)==len(after)
    assert [(a,b) for a,b in zip(before,after) if a!=b]==[
        ("progress['test_trajectories_completed'] != 67","progress['test_trajectories_completed'] != 58")]


def test_original_observer_numeric_code_and_full_case_guards_unchanged():
    config=current.read(current.ROOT/'configs/runtime/fm_cfm_conditional_GP6001104_full80_failure_observation.v1.json')
    job,model=current.verify(config)
    assert job['epochs']==600 and job['expected_optimizer_updates']==375000
    assert job['test_trajectories']==80 and job['test_points_per_trajectory']==500
    assert model['parameters']['target_mode']=='gaussian_conditional'
    assert config['settings']['minimum_commit_headroom_bytes']==30*1024**3
    assert config['solver_parameters']==dict(solver='dopri5',rtol=1e-7,atol=1e-9,max_num_steps=10000)
    old=current.previous.namespace();new=current.namespace()
    assert ast.dump(ast.parse(inspect.getsource(old['worker'])))==ast.dump(ast.parse(inspect.getsource(new['worker'])))
    assert new['RHSObserver'] is old['RHSObserver'] and new['verify'] is current.verify
