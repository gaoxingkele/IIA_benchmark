import ast
from copy import deepcopy
import inspect
import pytest
from scripts.flow_matching import run_pi_causal42_completed_full_replay_v1 as replay
from scripts.flow_matching import register_pi_causal42_completed_full_replay_v1 as registration


def test_only_seed_and_branch_predicates_change_in_existing_full_validator():
    old=ast.parse(inspect.getsource(replay.previous.verify))
    new=replay.verify_ast()
    old_compares=[ast.unparse(n) for n in ast.walk(old) if isinstance(n,ast.Compare)]
    new_compares=[ast.unparse(n) for n in ast.walk(new) if isinstance(n,ast.Compare)]
    changes=[(a,b) for a,b in zip(old_compares,new_compares) if a!=b]
    assert changes==[("job['seed'] != 1105","job['seed'] != 42"),
        ("job['branch'] != 'source_runtime'","job['branch'] != 'causal_axis_repaired'")]
    assert len(old_compares)==len(new_compares)


def test_registered_actual_completed_model_scope_scores_and_guards():
    config=replay.read('configs/runtime/fm_pi_causal42_completed_full_replay.v1.json')
    queue,native,review=replay.verify(config)
    assert len(queue['jobs'])==1140 and queue['settings']['seeds']==[42,1103,1104,1105,1106,1107]
    assert config['required_original_training_updates']==1554 and config['required_full_saved_score_streams']==15
    assert native['settings']==replay.read(config['native_template'])['settings']
    assert native['score_tolerance']==replay.read(config['native_template'])['score_tolerance']
    for key,value in [('required_original_training_updates',1553),('required_full_train_score_points',13238199),
        ('required_full_test_score_points',87799),('required_full_saved_score_streams',14)]:
        changed=deepcopy(config);changed[key]=value
        with pytest.raises(ValueError):replay.verify(changed)


def test_registration_changes_only_metadata_paths_and_preserves_full_stage_code():
    tree=registration.registration_ast()
    assert len(tree.body)==1 and tree.body[0].name=='main'
    constants=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'configs/experiments/fm_pi_causal42_full_incident_recovery_queue.v5.json' in constants
    assert not any('pi_source1105_completed_full_replay' in v for v in constants)
