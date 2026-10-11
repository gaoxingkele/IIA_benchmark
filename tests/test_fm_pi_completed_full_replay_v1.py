from copy import deepcopy

import pytest

from scripts.flow_matching.verify_pi_original_full_replay_completion_v1 import verify_terminal


def completed():
    shapes = dict(train_energy=[13238200], labels=[87800], threshold=[])
    completion = dict(original_id='original_seed42', result_sha256='original_result', original_parameters=1930289,
        original_full_score_shapes=shapes, full_native_budget=dict(epochs_completed=3, training_windows=132382),
        scored_test_points=87800, dropped_test_tail_points=41)
    report = dict(passed=True, full_native_checkpoint_reinference_passed=True,
        original_full_checkpoint_loaded_strictly=True, original_id='original_seed42',
        original_result_sha256='original_result', full_parameter_count=1930289, full_native_batch_size=256,
        cpu_threads=2, completed_original_training_seed_slots=1, additional_training_seed_slots=0,
        strict_TAB_result=False, author_equivalence_certified=False,
        all_original_score_streams_replayed={k:dict(shape=v, bitwise_equal=True, maximum_absolute_error=0.) for k,v in shapes.items()},
        rng=dict(fully_consumed_original_training_epochs=3, original_train_windows=132382,
                 original_batch_size=256, optimizer_updates_repeated=0),
        full_original_training_score_points=13238200, full_original_test_score_points=87800,
        dropped_original_test_tail_points=41)
    receipt = dict(exit_code=0, resource_guard_aborted=False, full_native_checkpoint_reinference_passed=True,
                   additional_training_seed_slots=0)
    return report, receipt, dict(cpu_threads=2), completion


def test_actual_complete_original_replay_contract():
    verify_terminal(*completed())


@pytest.mark.parametrize('key,value', [('exit_code',1),('resource_guard_aborted',True),('additional_training_seed_slots',1)])
def test_failed_or_extra_seed_replays_are_not_completions(key, value):
    items = list(completed())
    items[1][key] = value
    with pytest.raises(ValueError):
        verify_terminal(*items)


def test_missing_or_truncated_full_stream_is_not_replay():
    report, receipt, producer, completion = completed()
    missing = deepcopy(report)
    del missing['all_original_score_streams_replayed']['threshold']
    with pytest.raises(ValueError, match='Every original'):
        verify_terminal(missing,receipt,producer,completion)
    report['all_original_score_streams_replayed']['labels']['shape'] = [87700]
    with pytest.raises(ValueError, match='full replay'):
        verify_terminal(report,receipt,producer,completion)


@pytest.mark.parametrize('key,value', [('fully_consumed_original_training_epochs',2),
    ('original_train_windows',132381),('optimizer_updates_repeated',1)])
def test_original_sampler_draws_are_not_partial_or_retraining(key, value):
    items = list(completed())
    items[0]['rng'][key] = value
    with pytest.raises(ValueError, match='sampler'):
        verify_terminal(*items)


def test_nonidentical_replay_is_not_attested_bitwise_identical():
    items = list(completed())
    items[0]['all_original_score_streams_replayed']['train_energy']['maximum_absolute_error'] = 1e-7
    with pytest.raises(ValueError, match='bitwise identical'):
        verify_terminal(*items)
