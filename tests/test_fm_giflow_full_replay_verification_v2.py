import copy

import pytest

from scripts.flow_matching.verify_giflow_corrected_full_replay_v2 import full_coverage,check_terminal


def original(seed=394,targets=264837):
    job=dict(id='full_seed'+str(seed),seed=seed,track='reviewed_corrected',
        arguments=dict(batch_size=128,window=24,channel=1))
    result=dict(job=job,data_binding=dict(windows={'test':1747},nodes=36,
        evaluation_targets_in_input=0,timestamp_intersections=0),
        artifacts=[dict(targets=targets//14+(1 if i<targets%14 else 0)) for i in range(14)],
        full_test_batches=14,test_targets_with_window_repetitions=targets,epochs_completed=68,optimizer_updates=3264,
        selection=dict(test_used_in_validation=False,tested_weights_match_validation_checkpoint=True,
            actually_evaluated_weights_saved='evaluated_model.pt'),metrics={'mae':10.0})
    return job,result


def test_both_original_full_seed_masks_keep_all_targets_instead_of_copying_other_seed_count():
    for seed,targets in [(393,268613),(394,264837)]:
        job,result=original(seed,targets)
        coverage=full_coverage(result,job)
        assert coverage['full_masked_test_targets']==targets
        assert coverage['full_test_batches']==14 and coverage['full_test_windows']==1747
        assert coverage['observed_forward_calls']==280 and coverage['observed_window_evaluations']==34940
        assert coverage['actual_optimizer_updates']==3264


def test_partial_native_windows_batches_or_seed_specific_masks_are_rejected():
    job,result=original()
    for key,value in [('full_test_batches',13),('test_targets_with_window_repetitions',268613),
        ('artifacts',result['artifacts'][:-1])]:
        bad=copy.deepcopy(result)
        bad[key]=value
        with pytest.raises(ValueError): full_coverage(bad,job)
    for key,value in [('batch_size',64),('window',12),('channel',2)]:
        bad=copy.deepcopy(job)
        bad['arguments'][key]=value
        with pytest.raises(ValueError): full_coverage(result,bad)
    bad=copy.deepcopy(result)
    bad['data_binding']['windows']['test']=1000
    with pytest.raises(ValueError): full_coverage(bad,job)


def test_only_complete_native_terminal_replay_same_seed_budget_and_selection_can_pass():
    job,result=original()
    raw=dict(job_id=job['id'],seed=job['seed'],original_metrics=result['metrics'],
        full_checkpoint_contents_and_reinference_passed=True,full_data_masks_graph_scaler_and_splits_bitwise=True,
        actual_tested_weights_restored_bitwise=True,deterministic_model_forwards_consume_no_RNG=True,
        original_training_seed_slots_unchanged=True,not_a_new_training_seed=True,cuda_context_initialized=False,
        author_equivalence_certified=False,**{k:v for k,v in full_coverage(result,job).items()
            if k not in ['actual_full_training_epochs','actual_optimizer_updates','additional_training_seed_slots']})
    proof=dict(raw,selection=result['selection'],tested_checkpoint_selected_on_validation=True,
        source_test_block_unchanged=True,test_used_in_validation=False,additional_training_seed_slots=0,
        actual_full_training_epochs=68,actual_optimizer_updates=3264)
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_model_replay_passed=True)
    check_terminal(proof,raw,receipt,result,job)
    for key,value in [('full_masked_test_targets',268613),('observed_forward_calls',260),
        ('actual_optimizer_updates',3200),('actual_full_training_epochs',67),('additional_training_seed_slots',1),
        ('seed',393),('tested_checkpoint_selected_on_validation',False),('source_test_block_unchanged',False)]:
        bad=dict(proof,**{key:value})
        with pytest.raises(ValueError): check_terminal(bad,raw,receipt,result,job)
    for key,value in [('exit_code',1),('resource_guard_aborted',True),('full_model_replay_passed',False)]:
        with pytest.raises(ValueError): check_terminal(proof,raw,dict(receipt,**{key:value}),result,job)
