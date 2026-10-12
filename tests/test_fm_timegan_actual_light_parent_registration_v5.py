from scripts.flow_matching.timegan_native_common_v1 import ROOT, read


def test_actual_new_parent_keeps_all45_routes_and_original_native_replay_budgets():
    old=read(ROOT/'configs/runtime/fm_timegan_full45_completion_replay.v4.json')
    new=read(ROOT/'configs/runtime/fm_timegan_full45_completion_replay.v5.json')
    assert new['routes']==old['routes'] and len(new['routes'])==45
    for field in ('settings','resource_lock','required_checkpoints_each','required_evaluators_each',
                  'protocol_iterations_each','equality_gate','required_full_generated_windows',
                  'required_full_predictive_test_windows','required_full_discriminative_test_windows','original_scope'):
        assert new[field]==old[field]
    assert new['parent_candidate_discovery']=='metadata_only_original_full_numerical_validation_retained_in_guarded_native_worker'


def test_actual_second_handoff_preserved_every_scientific_child_and_attempt():
    new=read(ROOT/'configs/runtime/fm_timegan_full45_completion_replay.v5.json')
    handoff=read(ROOT/new['state_root']/'handoff.json')
    previous=read(ROOT/new['state_root']/'previous_status_at_verified_handoff.json')
    assert handoff['previous_owner']==new['previous_replay_owner']
    assert handoff['no_native_child_present'] and handoff['waiting_controller_only_stopped']
    assert not handoff['original_native_child_stopped']
    assert previous['active_pid'] is None and not previous['verified_original_ids'] and not previous['failed']
    assert previous['required_original_models']==previous['pending_original_models']==45
