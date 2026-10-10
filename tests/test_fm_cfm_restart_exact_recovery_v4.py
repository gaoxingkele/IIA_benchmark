from scripts.flow_matching.register_cfm_restart_exact_recovery_v4 import ROOT, read, validate_recovery


def test_restart_retries_keep_all_original_slots_and_exact_scientific_budgets():
    original = read(ROOT / 'configs/experiments/fm_cfm_ts_original_queue.v1.json')
    recovery = read(ROOT / 'configs/experiments/fm_cfm_ts_exact_recovery_queue.v4.json')
    validate_recovery(original, recovery)
    selected = [j for j in recovery['jobs'] if j['id'] in recovery['recovery_job_ids']]
    assert len(recovery['jobs']) == 285
    assert {(j['variant'], j['dataset'], j['seed']) for j in selected} == {
        ('node', 'pendulum', 1103), ('node', 'pendulum', 1104)}
    assert len(recovery['prior_full_resolutions']) == 3
    reasons = {e['reason'] for e in recovery['recovery_evidence']}
    assert reasons == {'authoritative_resource_guard_exit', 'preserved_preboot_partial_no_live_worker'}
    assert all(j['epochs'] == 10 and j['recovery_original_job_id'] for j in selected)
