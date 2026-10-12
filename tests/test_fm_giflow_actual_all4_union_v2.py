from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read

CONFIG='configs/reproducibility/fm_giflow_all4_completed_union.2026-10-12_v2.json'


def test_actual_full_four_model56_batch1120_forward_proofs_and_complete_target_counts():
    config=read(ROOT/CONFIG);proof=read(ROOT/config['output_proof']);terminal=read(ROOT/config['state_root']/'receipt.json')
    assert terminal['exit_code']==0 and terminal['all_completed_model_arrays_verified'] and not terminal['resource_guard_aborted']
    assert proof['original_canonical_slots']==140 and proof['completed_canonical_models']==4
    assert proof['full_scoring_batches']==56 and proof['full_test_window_evaluations']==6988
    assert proof['full_native_forward_calls']==1120 and proof['all_completed_canonical_models_replayed_and_independently_verified']
    assert {r['job_id'] for r in proof['records']}==set(config['expected_completed_ids'])
    assert proof['full_masked_target_evaluations']==sum(r['full_masked_targets'] for r in proof['records'])
    for record in proof['records']:
        assert record['native_full_checkpoint_replay_verified'] and record['complete_original_arrays_and_metrics_verified']
        assert len(record['batches'])==14 and record['full_native_forward_calls']==280


def test_all_original_array_guards_and_tolerances_and_five_seed_obligations_retained():
    config=read(ROOT/CONFIG);prior=read(ROOT/'configs/reproducibility/fm_giflow_complete_replay_union.2026-10-11_v1.json')
    follow=read(ROOT/'configs/reproducibility/fm_giflow_corrected_seed394_replay_followup.2026-10-11_v2.json')
    assert config['prediction_tolerance']==prior['prediction_tolerance']
    assert config['metric_tolerance']==prior['metric_tolerance']
    assert config['settings']==follow['settings'] and config['resource_lock']==follow['resource_lock']
    proof=read(ROOT/config['output_proof'])
    assert proof['additional_training_seed_slots']==proof['actual_new_training_seed_slots']==0
    assert proof['canonical_pending_training_slots']==136 and proof['no_model_reexecution']
    for group in proof['partial_seed_group_statistics'].values():
        assert group['required_seeds']==5 and group['completed_seeds']==[393,394]
        assert all(stats['n']==2 and stats['std'] is not None and stats['ci95_low'] is not None for stats in group['uncertainty'].values())
    assert not proof['author_equivalence_certified'] and not proof['all_original_experiments_complete']
