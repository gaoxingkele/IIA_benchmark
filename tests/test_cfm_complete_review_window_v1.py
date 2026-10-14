from scripts.flow_matching.reserve_cfm_complete_review_window_v1 import window_action, verify, ROOT, read


def test_no_commit_headroom_and_no_real_numeric_review_must_release():
    assert window_action(['slot'], [], False, False, True, 1, 900) == 'release_no_original_review_headroom'


def test_running_full_numeric_review_can_use_its_existing_admitted_memory():
    assert window_action(['slot'], [], False, True, True, 10, 900) == 'allow_original_full_reviews'


def test_failure_completion_emergency_and_window_deadline_release_only_lease():
    assert window_action([], [], True, False, True, 10, 900) == 'release_all_target_proofs_complete'
    assert window_action(['slot'], ['slot'], True, False, True, 10, 900) == 'release_unresolved_failure'
    assert window_action(['slot'], [], True, True, False, 10, 900) == 'release_emergency_headroom'
    assert window_action(['slot'], [], True, True, True, 900, 900) == 'release_window_deadline_only'


def test_registered_guards_family_lock_and_all_full_routes_are_identical():
    config = read(ROOT / 'configs/runtime/fm_cfm_complete_review_priority_window.v1.json')
    primary, supplement = verify(config)
    assert primary['settings'] == supplement['settings'] == config['review_settings']
    assert len(primary['routes']) == len(supplement['routes']) == 285
