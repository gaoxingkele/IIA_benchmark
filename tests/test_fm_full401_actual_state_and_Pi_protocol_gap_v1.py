from scripts.flow_matching import audit_full401_image_checkpoint_CPU_v1 as audit


def report():
    return audit.read(audit.ROOT/'projects/flow_matching_research/results/2026-10-12/full401_CPU_state_and_GPU_replay_mutex_v1/actual_full401_CPU_restore_shared_GPU_admission_and_Pi_source_metrics.json')


def test_actual_complete_native_model_optimizer_CPU_state_passes_without_GPU_or_performance_credit():
    summary=report();config=audit.read(audit.ROOT/'configs/runtime/fm_full401_image_checkpoint_CPU_audit.v1.json')
    proof=audit.read(audit.ROOT/summary['CPU_saved_state_proof'])
    assert proof['passed'] and proof['full_native_parameters']==45289539
    assert proof['completed_original_epochs']==401 and proof['verified_original_optimizer_updates']==45714
    assert proof['complete_epoch_receipts_verified']==401 and proof['remaining_original_epochs']==1599
    assert proof['all_model_and_StableAdamW_variables_bitwise_restored']
    assert proof['Python_NumPy_native_CPU_and_order_RNG_states_bitwise_restored']
    assert not proof['CUDA_initialized'] and not proof['CUDA_RNG_restoration_verified']
    assert proof['new_optimizer_updates_executed']==0 and proof['new_benchmark_results']==0
    archive=audit.read(audit.ROOT/'configs/reproducibility/fm_full401_checkpoint_immutable_archive.2026-10-12_v1.json')
    assert audit.sha(archive['immutable_full_state_archive']['path'])==config['checkpoint_receipt']['sha256']


def test_actual_both_replay_controllers_preserve_all_tasks_and_reject_occupied_training_GPU():
    summary=report();cases=summary['actual_two_idle_replay_handoffs_and_training_GPU_mutex_denial']
    assert len(cases)==2
    assert sorted(case['actual_state']['required_count'] for case in cases)==[49,413]
    for case in cases:
        assert case['actual_state']['verified_count']==0 and case['actual_state']['active_pid'] is None
        assert case['actual_denied_reservation']['blocked_lock'].endswith('fm_grasp_protocol_gpu.lock')
        assert case['GPU_work_executed'] is False


def test_verified_source_protocol_high_PA_F1_does_not_certify_strict_TAB_or_all1140_jobs():
    summary=report()
    assert summary['Pi_original_full_score_streams_independently_reread']==15
    assert summary['Pi_original_training_updates_verified']==1554
    assert summary['Pi_author_protocol_F1']>.98 and summary['Pi_same_threshold_no_point_adjustment_F1']<.03
    assert summary['Pi_raw_score_AUROC']<.5
    assert summary['Pi_source_threshold_uses_pooled_train_and_test_scores']
    assert summary['Pi_single_seed_source_protocol_not_strict_TAB'] and summary['Pi_paper_equivalence_not_certified']
    assert summary['Pi_remaining_original_full1140_scope_retained']
    assert summary['new_strict_TAB_leaderboard_results']==0 and not summary['all_original_experiments_complete']
