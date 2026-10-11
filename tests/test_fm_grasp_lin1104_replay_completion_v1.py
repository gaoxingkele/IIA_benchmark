import copy

import pytest

from scripts.flow_matching.verify_grasp_lin1104_full_replay_completion_v1 import check_terminal


def test_partial_wrong_model_or_failed_replay_cannot_be_completed():
    config = dict(full_job_count=1, full_results=[dict(canonical_id='original_lin_seed1104')])
    report = dict(full_job_count=1, full_checkpoint_contents_and_re_inference_passed=True,
        additional_training_seed_slots=0, cuda_context_initialized=False,
        records=[dict(canonical_id='original_lin_seed1104', epochs_completed=1500,
            optimizer_updates=93000, parameter_count=69244,
            full_checkpoint_restoration_and_reinference_passed=True)])
    receipt = dict(exit_code=0, resource_guard_aborted=False, full_replay_passed=True)
    check_terminal(report, receipt, config)
    for key, value in [('epochs_completed', 1499), ('optimizer_updates', 92999),
                       ('parameter_count', 1000), ('canonical_id', 'another_seed')]:
        bad = copy.deepcopy(report)
        bad['records'][0][key] = value
        with pytest.raises(ValueError):
            check_terminal(bad, receipt, config)
    for key, value in [('exit_code', 1), ('resource_guard_aborted', True), ('full_replay_passed', False)]:
        bad = dict(receipt, **{key: value})
        with pytest.raises(ValueError):
            check_terminal(report, bad, config)
    for key, value in [('additional_training_seed_slots', 1), ('cuda_context_initialized', True)]:
        with pytest.raises(ValueError):
            check_terminal(dict(report, **{key: value}), receipt, config)
