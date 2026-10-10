import copy
import pytest

from scripts.flow_matching.audit_giflow_released_seed_checkpoint_v1 import released_boundary


def inputs():
    proof = dict(job_id='native_seed394', seed=394, full_test_windows=1747, full_test_batches=14,
                 full_masked_test_targets=264837, observed_forward_calls=280, observed_window_evaluations=34940,
                 full_checkpoint_contents_and_reinference_passed=True)
    result = dict(job=dict(id='native_seed394', seed=394, track='released_mirror'),
                  selection=dict(test_used_in_validation=True), data_binding=dict(windows=dict(test=1747)),
                  full_test_batches=14, test_targets_with_window_repetitions=264837,
                  epochs_completed=196, optimizer_updates=9408)
    return proof, result


def test_metadata_uses_actual_seed_targets_preserving_raw_proof():
    proof, result = inputs()
    old = copy.deepcopy(proof)
    reviewed = released_boundary(proof, result)
    assert proof == old
    assert '264837' in reviewed['boundary'] and '268613' not in reviewed['boundary']
    assert reviewed['additional_training_seed_slots'] == 0
    assert reviewed['actual_optimizer_updates'] == 9408


@pytest.mark.parametrize('field,value', [('seed',393), ('full_masked_test_targets',268613),
    ('full_test_batches',13), ('observed_forward_calls',260), ('observed_window_evaluations',30720)])
def test_rejects_other_seed_or_partial_coverage(field, value):
    proof, result = inputs()
    proof[field] = value
    with pytest.raises(ValueError):
        released_boundary(proof, result)
