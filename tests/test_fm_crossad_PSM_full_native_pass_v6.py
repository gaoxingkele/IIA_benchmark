"""Verify the actual complete PSM native proof, without adding seed credit."""
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.register_crossad_PSM_attention_completion_v3 import CONFIG,NATIVE,POST,PREVIOUS


def test_actual_entire_original_PSM_model_all_points_and_author_evaluators_replayed():
    native=full.read(NATIVE)
    job=next(j for j in full.read(native['queue'])['jobs'] if j['id']==native['job_id'])
    proof,terminal=full.native_terminal(native,job)
    assert terminal['exit_code']==0 and terminal['full_native_model_replay_passed'] and not terminal['resource_guard_aborted']
    assert proof['full_model_parameters']==1328459
    assert proof['full_score_points']==dict(train=105984,validation=26497,test=87841)
    assert proof['original_native_test_points']==87744
    assert proof['full_original_model_reinference_passed'] and proof['full_original_author_metric_reexecution_passed']
    assert proof['original_SPOT_init_model_inference_and_threshold_reexecution_passed']
    assert proof['original_full_state_dictionary_loaded_strictly']
    assert proof['attention_scores_released_only_after_last_original_softmax_use']


def test_same_batch_source_guards_and_tolerances_without_overwriting_failures():
    config=full.read(CONFIG);native=full.read(NATIVE);post=full.read(POST)
    prior=full.read(full.read(PREVIOUS)['native_configuration'])
    for key in full.GUARDS:assert native[key]==prior[key]==post[key]
    failed=full.read(prior['state_root']+'/receipt.json')
    passed=full.read(native['state_root']+'/receipt.json')
    assert failed['resource_guard_aborted'] and not failed['full_native_model_replay_passed']
    assert failed['peak_tree_rss_bytes']>prior['settings']['maximum_worker_rss_bytes']
    assert passed['peak_tree_rss_bytes']<=native['settings']['maximum_worker_rss_bytes']
    assert passed['peak_tree_private_bytes']<=native['settings']['maximum_worker_private_bytes']
    job=next(j for j in full.read(native['queue'])['jobs'] if j['id']==native['job_id'])
    assert job['train_parameters']['batch_size']==128 and job['model_parameters']['seq_len']==192
    assert config['metric_tolerance']==dict(rtol=1e-12,atol=1e-12) and config['score_tolerance']==full.SCORE_TOLERANCE


def test_a_fixed_release_native_proof_adds_no_training_seed_or_TAB_or_paper_equivalence():
    native=full.read(NATIVE);proof=full.read(full.ROOT/native['artifact_directory']/'full_native_model_replay.json')
    assert proof['fixed_release_checkpoint_evaluation']
    assert proof['additional_training_seed_slots']==proof['completed_training_seed_slots']==0
    assert proof['required_original_cohort_training_seeds']==0
    assert not proof['strict_TAB_result'] and not proof['author_equivalence_certified']
    assert not proof['all_original_experiments_complete'] and not proof['cuda_context_initialized']
