import copy
import pytest

from scripts.flow_matching.followup_giflow_seed_replay_verification_v1 import (
    successful_terminal,independent_success,exact_producer_live,clarified_review)

def test_only_actual_successful_full_native_terminal_allows_followup():
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_model_replay_passed=True)
    successful_terminal(receipt)
    for key,value in [('exit_code',1),('resource_guard_aborted',True),('full_model_replay_passed',False)]:
        with pytest.raises(ValueError):
            successful_terminal(dict(receipt,**{key:value}))
    with pytest.raises(ValueError): successful_terminal({})

def test_completion_requires_actual_independent_full_coverage_contract():
    proof=dict(full_model_replay_and_independent_saved_arrays_verified=True,seed=394,additional_training_seed_slots=0,
        coverage=dict(full_test_windows=1747,full_test_batches=14,full_masked_test_targets=264837,observed_forward_calls=280))
    assert independent_success(proof)
    for key,value in [('seed',393),('additional_training_seed_slots',1),('full_model_replay_and_independent_saved_arrays_verified',False)]:
        assert not independent_success(dict(proof,**{key:value}))
    for key,value in [('full_test_windows',100),('full_test_batches',13),('full_masked_test_targets',268613),('observed_forward_calls',260)]:
        assert not independent_success(dict(proof,coverage=dict(proof['coverage'],**{key:value})))
    assert not independent_success({'passed':True})

def test_same_pid_requires_matching_creation_time_module_and_configuration():
    config=dict(expected_producer_pid=12,expected_producer_create_time=100.,producer_module='full.native',replay_config='frozen.json')
    class Process:
        def __init__(self,pid): assert pid==12
        def create_time(self): return 100.
        def cmdline(self): return ['python','-m','full.native','--config','frozen.json']
    assert exact_producer_live(config,Process)
    for key,value in [('expected_producer_create_time',99.),('producer_module','other.native'),('replay_config','other.json')]:
        assert not exact_producer_live(dict(config,**{key:value}),Process)

def test_seed_target_wording_correction_never_changes_arrays_metrics_or_coverage():
    result=dict(job={'id':'corrected394'},test_targets_with_window_repetitions=264837,_raw_sha256='raw')
    raw=dict(job_id='corrected394',seed=394,full_checkpoint_contents_and_reinference_passed=True,
        full_masked_test_targets=264837,full_batch_receipts=[{'batch':i} for i in range(14)],metrics={'mse':458.})
    reviewed=dict(copy.deepcopy(raw),original_raw_replay_proof_sha256='raw',boundary='all1747 windows/268613 masked targets/280 forwards')
    corrected=clarified_review(raw,reviewed,result)
    assert corrected['boundary']=='all1747 windows/264837 masked targets/280 forwards'
    assert {k:v for k,v in corrected.items() if k!='boundary'}=={k:v for k,v in reviewed.items() if k!='boundary'}
    for key,value in [('full_masked_test_targets',268613),('metrics',{'mse':450.}),('seed',393),('original_raw_replay_proof_sha256','other')]:
        with pytest.raises(ValueError): clarified_review(raw,dict(reviewed,**{key:value}),result)
