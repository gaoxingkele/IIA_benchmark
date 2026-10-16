from copy import deepcopy
import pytest
from scripts.flow_matching.otcfm_singlecell_result_integrity_v2 import validate_record
from scripts.flow_matching.run_otcfm_singlecell_native_v4 import identity
from scripts.flow_matching.otcfm_singlecell_native_contract_v1 import scientific_cases


def fixture():
    job = scientific_cases()[0]
    job['data'] = dict(days={'2': 7264, '3': 8544, '4': 10933, '7': 6960})
    recipe = job['scientific_protocol']
    result = dict(status='completed', job=deepcopy(job), job_sha256=identity(job), maximum_epochs=1000,
        batch_size=512, parameter_count=9093, full_timepoint_cell_counts=job['data']['days'], original_loader_whiten_true=True,
        epoch_history=[dict(epoch=i, optimizer_updates=10*i) for i in range(1, 1001)], actual_completed_epochs=1000,
        optimizer_updates=7200, best_checkpoint='F:/evidence/epoch_0719.ckpt',
        validation_history=[dict(epoch=720, finite=True, validation_loss=.1), dict(epoch=1000, finite=True, validation_loss=.2)], early_stopping_epoch=0, training_seconds=500., termination_reason='maximum_epochs',
        early_stopping_patience_checks=100, test_metrics={'test/t_out/1-Wasserstein': 1.}, primary_heldout_W1=1.,
        historical_paper_equivalence_certified=False, strict_TAB_result=False)
    return dict(primary_metric='test/t_out/1-Wasserstein'), job, result, dict(passed=True, exit_code=0, resource_guard_aborted=False)


def test_exact_full_budget_record_is_only_a_provenance_fixture_not_benchmark_performance():
    validate_record(*fixture())


@pytest.mark.parametrize('fault', ['job', 'guard', 'guard_abort', 'data', 'parameters', 'batch', 'epoch_gap',
    'updates', 'posttest_counter', 'wrong_selected_epoch', 'shortened', 'patience', 'nonfinite', 'primary', 'paper_claim', 'TAB_claim'])
def test_partial_changed_or_misattributed_results_are_rejected(fault):
    q, j, r, g = fixture()
    if fault == 'job': r['job']['seed'] = 999
    elif fault == 'guard': g['exit_code'] = 1
    elif fault == 'guard_abort': g['resource_guard_aborted'] = True
    elif fault == 'data': r['full_timepoint_cell_counts'] = {'2': 100}
    elif fault == 'parameters': r['parameter_count'] = 100
    elif fault == 'batch': r['batch_size'] = 128
    elif fault == 'epoch_gap': r['epoch_history'][20]['epoch'] = 1
    elif fault == 'updates': r['epoch_history'][-1]['optimizer_updates'] -= 1
    elif fault == 'posttest_counter': r['optimizer_updates'] = 10000
    elif fault == 'wrong_selected_epoch': r['best_checkpoint'] = 'F:/evidence/epoch_0999.ckpt'
    elif fault == 'shortened':
        r['epoch_history'].pop(); r['actual_completed_epochs'] = 999; r['optimizer_updates'] = 9990
    elif fault == 'patience': r['early_stopping_patience_checks'] = 3
    elif fault == 'nonfinite': r['test_metrics']['test/other'] = float('nan')
    elif fault == 'primary': r['primary_heldout_W1'] = .9
    elif fault == 'paper_claim': r['historical_paper_equivalence_certified'] = True
    else: r['strict_TAB_result'] = True
    with pytest.raises(ValueError): validate_record(q, j, r, g)


def test_posttest_checkpoint_counter_cannot_replace_total_actual_training_updates():
    q, j, r, g = fixture()
    validate_record(q, j, r, g)
    assert r['optimizer_updates'] == 7200 and r['epoch_history'][-1]['optimizer_updates'] == 10000


@pytest.mark.parametrize('cap_reached', [True, False])
def test_only_real_registered_wall_time_cap_permits_last_partial_epoch(cap_reached):
    from scripts.flow_matching.otcfm_singlecell_native_contract_v1 import protocol
    q, j, r, g = fixture()
    j['scientific_protocol'] = protocol('paper_declared')
    r.update(job=deepcopy(j), job_sha256=identity(j), batch_size=128, actual_completed_epochs=103,
        epoch_history=[dict(epoch=i, optimizer_updates=43*i) for i in range(1, 104)],
        best_checkpoint='F:/evidence/epoch_0099.ckpt', optimizer_updates=4300,
        validation_history=[dict(epoch=100, finite=True, validation_loss=.1)],
        termination_reason='native_paper_wall_time_cap', training_seconds=6000. if cap_reached else 5999.,
        early_stopping_patience_checks=3)
    r['epoch_history'][-1]['optimizer_updates'] -= 10
    if cap_reached: validate_record(q, j, r, g)
    else:
        with pytest.raises(ValueError): validate_record(q, j, r, g)
