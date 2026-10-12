from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.register_crossad_PSM_lifetime_completion_v1 import CONFIG,NATIVE,POST


def test_original_batch_all_points_weights_and_all_guards_retained():
    config=full.read(CONFIG);native=full.read(NATIVE);post=full.read(POST)
    previous=full.read('configs/reproducibility/fm_crossad_recovery_completion_children_v1/crossad_author__release_checkpoint__psm__seed2025__resource_recovery_v1_native.json')
    for name in full.GUARDS:assert native[name]==previous[name]==post[name]
    assert native['queue']==previous['queue'] and native['job_id']==previous['job_id']
    assert native['independent_point_completion']==previous['independent_point_completion']
    job=next(j for j in full.read(native['queue'])['jobs'] if j['id']==native['job_id'])
    assert job['train_parameters']['batch_size']==128 and job['model_parameters']['seq_len']==192
    result=full.read(full.ROOT/job['output_directory']/'result.json')
    assert result['details']['parameter_count']==1328459
    assert result['details']['native_points']==dict(train=105984,validation=26497,test=87841)
    assert config['required_original_jobs']==139 and config['original_scope']==full.SCOPE
    assert config['score_tolerance']==full.SCORE_TOLERANCE and config['metric_tolerance']==full.METRIC_TOLERANCE


def test_actual_point_certificate_reused_and_failed_native_attempt_preserved():
    config=full.read(CONFIG);native=full.read(NATIVE)
    point=full.read(native['independent_point_completion']);terminal=full.read(config['point_terminal'])
    assert point['passed'] and point['full_raw_labels_and_source_rows_checked']
    assert terminal['exit_code']==0 and terminal['independent_stage_passed'] and not terminal['resource_guard_aborted']
    old=full.read('configs/reproducibility/fm_crossad_recovery_completion_children_v1/crossad_author__release_checkpoint__psm__seed2025__resource_recovery_v1_native.json')
    failed=full.read(full.ROOT/old['state_root']/'receipt.json')
    assert failed['exit_code']==1 and failed['resource_guard_aborted'] and not failed['full_native_model_replay_passed']
    assert native['state_root']!=old['state_root'] and native['artifact_directory']!=old['artifact_directory']


def test_actual_numeric_diagnostics_passed_without_becoming_full_PSM_completion():
    config=full.read(CONFIG);terminal=full.read(config['diagnostic_terminal'])
    assert terminal['exit_code']==0 and terminal['passed'] and not terminal['resource_guard_aborted']
    assert terminal['diagnostics_are_not_original_full_PSM_replay'] and not terminal['benchmark_performance']
