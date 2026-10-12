from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.register_timegan_checkpointed_sine_recovery_v4 import RECOVERY, REPLAY


def test_all45_cases_original_full23_checkpoints_and20_evaluators_are_retained():
    old = read(ROOT/'configs/runtime/fm_timegan_full45_completion_replay.v3.json')
    new = read(ROOT/REPLAY)
    assert len(new['routes']) == len(old['routes']) == 45
    assert [r['canonical_job'] for r in new['routes']] == [r['canonical_job'] for r in old['routes']]
    assert new['required_checkpoints_each'] == old['required_checkpoints_each'] == 23
    assert new['required_evaluators_each'] == old['required_evaluators_each'] == 20
    assert new['protocol_iterations_each'] == old['protocol_iterations_each'] == 70000
    assert new['equality_gate'] == old['equality_gate'] == 'bitwise'
    for old_route, new_route in zip(old['routes'],new['routes']):
        if old_route['canonical_job']['id'] != 'timegan_native__gru__sine__seed1103':
            assert new_route == old_route
        else:
            assert new_route['recovery_configuration'] == RECOVERY
            assert new_route['full_original_data_shape'] == [10000,24,5]
    assert new['settings'] == old['settings'] and new['resource_lock'] == old['resource_lock']
    assert new['original_scope'] == dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)


def test_full50000_native_recovery_uses_same_two_actual_pretraining_states():
    old = read(ROOT/'configs/reproducibility/fm_timegan_sine_stage_recovery.2026-10-12_v3.json')
    new = read(ROOT/RECOVERY)
    queue = read(ROOT/new['original_queue'])
    assert new['canonical_job'] == old['canonical_job']
    assert new['stage_receipt_sha256'] == old['stage_receipt_sha256']
    assert new['native_restore_audit_sha256'] == old['native_restore_audit_sha256']
    assert new['joint_iterations'] == 50000 and new['joint_checkpoint_interval'] == 1000
    assert new['checkpoint_full_variable_count'] == 284
    assert new['original_training_settings'] == queue['settings'] == old['original_training_settings']
    assert new['output_directory'] != old['output_directory']
    assert new['full_scope_original_jobs'] == 45 and new['independent_training_seed_slots_added'] == 0
    for tag in ('embedding_complete','supervised_complete'):
        receipt = Path(new['canonical_job']['output_directory'])/'checkpoints'/tag/'receipt.json'
        assert sha(receipt) == new['stage_receipt_sha256'][tag]


def test_actual_handoff_stopped_only_exact_idle_owner_without_native_children():
    new = read(ROOT/REPLAY)
    handoff = read(ROOT/new['state_root']/'handoff.json')
    identity = read(ROOT/new['state_root']/'handoff_identity.json')
    previous = read(ROOT/new['state_root']/'previous_status_at_verified_handoff.json')
    assert handoff['previous_owner'] == new['previous_replay_owner']
    assert handoff['waiting_controller_only_stopped'] and handoff['no_native_child_present']
    assert not handoff['original_native_child_stopped']
    assert identity['no_native_child_present'] and identity['only_controller_will_be_stopped']
    assert previous['active_pid'] is None and not previous['verified_original_ids'] and not previous['failed']
    assert previous['required_original_models'] == previous['pending_original_models'] == 45
