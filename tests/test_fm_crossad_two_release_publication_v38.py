import copy
import pytest
from scripts.flow_matching.publish_crossad_two_release_metrics_v38 import verified_release


def test_partial_or_fresh_training_never_fills_fixed_release_slots():
    point=dict(passed=True,fixed_release_checkpoint_evaluation=True,completed_training_seed_slots=0,training_epochs_completed=0,job_id='released',full_native_points={'train':100,'validation':25,'test':80})
    model=dict(full_original_model_reinference_passed=True,full_original_author_metric_reexecution_passed=True,
        original_full_state_dictionary_loaded_strictly=True,original_SPOT_init_model_inference_and_threshold_reexecution_passed=True,
        fixed_release_checkpoint_evaluation=True,completed_training_seed_slots=0,registered_required_release_evaluations=1,
        additional_training_seed_slots=0,cuda_context_initialized=False,job_id='released',full_score_points=point['full_native_points'])
    guard=dict(exit_code=0,resource_guard_aborted=False,full_native_model_replay_passed=True)
    verified_release(point,model,guard)
    for field,value in [('completed_training_seed_slots',1),('fixed_release_checkpoint_evaluation',False),('full_original_model_reinference_passed',False),('original_SPOT_init_model_inference_and_threshold_reexecution_passed',False),('registered_required_release_evaluations',6),('full_score_points',{'train':100,'validation':25,'test':79})]:
        changed=copy.deepcopy(model);changed[field]=value
        with pytest.raises(ValueError):verified_release(point,changed,guard)
    changed=copy.deepcopy(guard);changed['resource_guard_aborted']=True
    with pytest.raises(ValueError):verified_release(point,model,changed)
