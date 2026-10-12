from scripts.flow_matching import run_pi_source1103_completed_full_replay_v2 as pipeline

CONFIG='configs/runtime/fm_pi_source1103_completed_full_replay.v2.json'


def test_actual_finished_full_original_seed_bound_with_all1140_jobs_and_original_guards():
    config=pipeline.read(CONFIG);queue,native_template,_=pipeline.verify(config)
    assert len(queue['jobs'])==1140 and queue['settings']['seeds']==[42,1103,1104,1105,1106,1107]
    case=config['cases'][0];point=pipeline.read(case['point_configuration']);native=pipeline.read(case['native_configuration'])
    for key in pipeline.original.GUARDS:assert native[key]==native_template[key]
    receipt=pipeline.read(point['state_root']+'/receipt.json')
    assert receipt['exit_code']==0 and receipt['full_original_point_audit_passed'] and not receipt['resource_guard_aborted']


def test_actual_all15_complete_score_streams_full_budget_and_raw_PA_metrics_kept_distinct():
    config=pipeline.read(CONFIG);point=pipeline.read(config['cases'][0]['point_configuration'])
    proof=pipeline.read(point['output_directory']+'/independent_full_recovery_completion.json')
    assert proof['passed'] and proof['original_id']==pipeline.CANONICAL and proof['seed']==1103
    assert proof['original_parameters']==1930289 and proof['actual_active_layers']==1
    assert proof['full_native_budget']['optimizer_minibatches']==1554
    assert proof['full_native_budget']['epochs_completed']==3
    assert len(proof['original_full_score_shapes'])==15 and proof['all_posthoc_controls_recomputed']==15
    assert proof['original_full_score_shapes']['train_energy']==[13238200]
    assert proof['scored_test_points']==87800 and proof['dropped_test_tail_points']==41
    assert proof['full_original_raw_labels_verified'] and proof['original_source_and_artifacts_verified']
    assert proof['independently_recomputed_metrics']['f1']>.9
    assert proof['independently_recomputed_metrics']['pointwise_no_PA_same_author_threshold']['f1']<.1
    assert proof['confusion_no_PA']['fp']==proof['confusion_author_PA']['fp']
    assert proof['completed_original_training_seed_slots']==1 and proof['retries_are_not_additional_training_seeds']
    assert not proof['strict_TAB_result'] and not proof['author_equivalence_certified'] and not proof['all_original_experiments_complete']
