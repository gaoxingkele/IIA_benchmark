from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.register_crossad_PSM_attention_completion_v3 import CONFIG,NATIVE,POST,PREVIOUS,DIAGNOSTIC


def test_same_full_source_PSM_weights_and_all_original_guards_and_tolerances():
    config=full.read(CONFIG);previous=full.read(PREVIOUS);native=full.read(NATIVE);post=full.read(POST)
    old=full.read(previous['native_configuration'])
    for key in full.GUARDS:assert native[key]==old[key]==post[key]
    for key in ('queue','job_id','independent_point_completion'):assert native[key]==old[key]
    for key in ('required_original_jobs','original_scope','score_tolerance','metric_tolerance','point_terminal'):
        assert config[key]==previous[key]
    assert config['required_original_jobs']==139 and config['metric_tolerance']=={'rtol':1e-12,'atol':1e-12}
    job=next(j for j in full.read(native['queue'])['jobs'] if j['id']==native['job_id'])
    assert job['train_parameters']['batch_size']==128 and job['model_parameters']['seq_len']==192
    result=full.read(full.ROOT/job['output_directory']/'result.json')
    assert result['details']['parameter_count']==1328459
    assert result['details']['native_points']==dict(train=105984,validation=26497,test=87841)


def test_actual_previous_failed_full_attempt_kept_with_a_new_version_not_identical_retry():
    previous=full.read(PREVIOUS);old=full.read(previous['native_configuration']);native=full.read(NATIVE)
    failed=full.read(old['state_root']+'/receipt.json')
    assert failed['resource_guard_aborted'] and not failed['full_native_model_replay_passed']
    assert failed['peak_tree_rss_bytes']>old['settings']['maximum_worker_rss_bytes']
    assert native['state_root']!=old['state_root'] and native['artifact_directory']!=old['artifact_directory']
    assert 'src/iia_benchmark/models/fm_crossad_attention_lifetime_v2.py' in {p['path'] for p in native['source_receipts']}


def test_actual_all_sixteen_guarded_numerical_checks_passed_without_dataset_completion_credit():
    config=full.read(CONFIG);terminal=full.read(config['diagnostic_terminal']);diagnostic=full.read(DIAGNOSTIC)
    assert terminal['passed'] and terminal['exit_code']==0 and not terminal['resource_guard_aborted']
    assert terminal['diagnostic_checks']==diagnostic['required_checks']==16
    assert terminal['diagnostics_are_not_original_full_PSM_replay'] and not terminal['benchmark_performance']
    assert terminal['original_replay_settings']==full.read(NATIVE)['settings']
    assert terminal['configuration_sha256']==full.digest(DIAGNOSTIC)
    for pin in config['source_receipts']:assert full.digest(pin['path'])==pin['sha256']
