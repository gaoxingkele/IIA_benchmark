"""Readiness must retain failed outcomes and await every other full certificate."""
import ast
import pytest
from scripts.flow_matching import run_cfm_gp_budget_outcome_completion_v2 as module


def complete_records():
    return [dict(original_canonical_id=i) for pair in module.planned.identities() for i in pair if i != module.FAILED]


def test_waits_for_every_nonfailed_certificate():
    records = complete_records()
    selected, missing = module.select_outcomes(records[:-1], module.planned.identities())
    assert selected is None and missing == [records[-1]['original_canonical_id']]


def test_nine_certificates_and_preserved_failed_slot_are_ready_for_outcome_audit():
    selected, missing = module.select_outcomes(complete_records(), module.planned.identities())
    assert selected is True and missing == []


def test_preserved_failed_seed_cannot_be_silently_replaced():
    with pytest.raises(ValueError):
        module.select_outcomes(complete_records() + [dict(original_canonical_id=module.FAILED)], module.planned.identities())


def test_duplicate_canonical_certificate_rejected():
    records = complete_records()
    with pytest.raises(ValueError): module.select_outcomes(records + records[:1], module.planned.identities())


def test_scope_cannot_drop_failed_pair():
    with pytest.raises(ValueError): module.select_outcomes(complete_records(), module.planned.identities()[1:])


def test_metadata_labels_do_not_claim_ten_successful_scores():
    tree = module.metadata_main_ast()
    source = ast.unparse(tree)
    assert 'required_original_model_outcomes=10' in source
    assert 'completed_all_original_full_outcomes_with_preserved_failure' in source
    assert 'required_complete_models' not in source
    assert 'model_training_executed=False' in source


def test_only_idle_failed_metric_waiter_can_be_handed_off():
    state = dict(state='waiting_every_original_full_seed_pair', required_complete_models=10,
        missing_complete_model_ids=[module.FAILED], active_pid=None, shared_lock_owned=False)
    assert module.idle_handoff_allowed('original', state, True, [])
    assert not module.idle_handoff_allowed('native', state, False, [])
    assert not module.idle_handoff_allowed('original', dict(state, active_pid=123), False, [])
    assert not module.idle_handoff_allowed('original', dict(state, shared_lock_owned=True), False, [])
    assert not module.idle_handoff_allowed('original', dict(state, state='auditing_all_ten_full_models'), False, [])
    assert not module.idle_handoff_allowed('original', dict(state, missing_complete_model_ids=[]), False, [])
    assert not module.idle_handoff_allowed('original', state, False, [dict(name='python.exe', exe='C:/Python/python.exe')])
