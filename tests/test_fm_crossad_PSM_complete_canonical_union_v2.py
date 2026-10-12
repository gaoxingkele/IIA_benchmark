"""Actual full-model/post proof and exact canonical credit, never seed credit."""
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching import run_crossad_recovery_completion_handoff_v1 as recovery

POST='configs/reproducibility/fm_crossad_PSM_independent_metric_keywords.2026-10-12_v7.json'
RUNTIME='configs/runtime/fm_crossad_recovery_completion_post_API_fix.v2.json'
IDENTITY='crossad_author__release_checkpoint__psm__seed2025'


def test_actual_original_PSM_scores_labels_rows_author_and_three_validation_operating_points_verified():
    post=full.read(POST);native=full.read(post['native_configuration'])
    job=next(j for j in full.read(native['queue'])['jobs'] if j['id']==native['job_id'])
    completed=recovery.completed_review(POST,job,post)
    proof=full.read(completed['independent_completed_replay_review'])
    assert proof['passed'] and proof['full_native_model_replay_passed']
    for key in ('all_original_raw_point_and_row_identities_independently_reviewed',
            'all_full_original_and_replayed_scores_independently_compared','all_original_author_metrics_reexecuted_and_compared'):
        assert proof[key]
    assert proof['full_point_counts']==dict(train=105984,validation=26497,test=87841)
    assert set(proof['replayed_validation_only_metrics_independently_recomputed'])=={'0.1','1','5'}
    assert proof['original_training_seed_slots_completed']==proof['additional_training_seed_slots']==0
    assert not proof['strict_TAB_result'] and not proof['author_equivalence_certified']


def test_recovery_alias_is_exact_original_science_and_full_post_completion_not_a_new_seed():
    runtime=full.read(RUNTIME);base=full.read(runtime['original_completion_configuration'])
    jobs={j['id']:j for j in full.read(base['original_queue'])['jobs']}
    route=next(r for r in runtime['recovery_routes'] if r['canonical_id']==IDENTITY)
    job=recovery.exact_route(route,jobs[IDENTITY],full.read(route['native_queue']),full.read(base['original_queue']))
    entry=full.read(full.ROOT/runtime['state_root']/'completed'/(IDENTITY+'.json'))
    assert entry['recovery_job_id']==job['id'] and entry['canonical_id']==IDENTITY
    assert entry['post_review_configuration']==POST and entry['additional_training_seed_slots']==0
    assert entry['completed_original_training_seed_slots']==0


def test_canonical_union_preserves_three_prior_certificates_and_entire139_scope():
    runtime=full.read(RUNTIME);state=full.read(runtime['state_root']+'/status.json')
    base=full.read(runtime['original_completion_configuration'])
    prior={x['job_id'] for x in base['preserved_verified_jobs']}
    assert prior|{IDENTITY}<=set(state['verified_original_ids'])
    assert state['required_original_jobs']==139 and state['pending_original_jobs']==139-len(state['verified_original_ids'])
    assert state['additional_training_seed_slots']==0 and IDENTITY in state['verified_exact_recovery_ids']
    assert not state['all_original_experiments_complete']


def test_both_actual_cold_handoffs_stopped_only_verified_idle_parents():
    cfg=full.read('configs/runtime/fm_crossad_full139_post_API_fix_handoff.v2.json')
    for role in ('original','recovery'):
        route=cfg['roles'][role];proof=full.read(route['handoff_state_root']+'/handoff.json')
        snapshot=full.read(route['handoff_state_root']+'/previous_status_at_verified_handoff.json')
        assert proof['waiting_controller_only_stopped'] and proof['no_native_child_present']
        assert not proof['original_native_child_stopped'] and snapshot['active_pid'] is None
        assert snapshot['verified_original_ids']==sorted(x['job_id'] for x in full.read(
            'configs/runtime/fm_crossad_completion_replay_queue.v1.json')['preserved_verified_jobs'])
