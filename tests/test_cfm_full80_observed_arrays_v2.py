import ast
import inspect
from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as current


def test_only_observed_training_metadata_schema_key_changes():
    tree = current.corrected_audit_ast()
    for node in ast.walk(tree):
        if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                and node.value.id == 'source' and isinstance(node.slice,ast.Constant)
                and node.slice.value == 'observed_states'):
            node.slice.value = 'train_states'
    assert ast.dump(tree) == ast.dump(ast.parse(inspect.getsource(current.previous.audit)))


def test_actual_frozen_full80_arrays_preserve_original_failure_and_every_coordinate():
    config = current.previous.native.read(current.previous.ROOT / 'configs/reproducibility/fm_cfm_full80_observed_arrays.2026-10-14_v2.json')
    report = current.audit(config)
    assert report['independent_checked_grid_coordinates'] == 40000
    assert report['attempted_original_trajectories'] == 80
    assert report['finite_full_grid_trajectories'] == 79 and report['failed_trajectory_ids'] == [103]
    assert report['full_original_MSE'] is None and not report['original_scoring_certified_complete']
    failed = report['failure_evidence'][0]
    assert failed['initial_state_inside_observed_training_state_range']
    assert failed['finite_state_at_first_nonfinite_velocity'] and failed['finite_velocity_near_float32_limit']
    assert not failed['learned_training_failure_root_cause_certified']
