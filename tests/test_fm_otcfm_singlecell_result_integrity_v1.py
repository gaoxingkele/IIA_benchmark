from copy import deepcopy
import pytest
from scripts.flow_matching.otcfm_singlecell_result_integrity_v1 import validate_record
from scripts.flow_matching.run_otcfm_singlecell_native_v4 import identity
from scripts.flow_matching.otcfm_singlecell_native_contract_v1 import scientific_cases


def fixture():
    job = scientific_cases()[0]
    job['data'] = dict(days={'2': 7264, '3': 8544, '4': 10933, '7': 6960})
    recipe = job['scientific_protocol']
    result = dict(status='completed', job=deepcopy(job), job_sha256=identity(job), maximum_epochs=1000,
        batch_size=512, parameter_count=9093, full_timepoint_cell_counts=job['data']['days'], original_loader_whiten_true=True,
        epoch_history=[dict(epoch=i, optimizer_updates=10*i) for i in range(1, 1001)], actual_completed_epochs=1000,
        optimizer_updates=10000, early_stopping_epoch=0, training_seconds=500., termination_reason='maximum_epochs',
        early_stopping_patience_checks=100, test_metrics={'test/t_out/1-Wasserstein': 1.}, primary_heldout_W1=1.,
        historical_paper_equivalence_certified=False, strict_TAB_result=False)
    return dict(primary_metric='test/t_out/1-Wasserstein'), job, result, dict(passed=True, exit_code=0, resource_guard_aborted=False)


def test_exact_full_budget_record_is_only_a_provenance_fixture_not_benchmark_performance():
    validate_record(*fixture())


@pytest.mark.parametrize('fault', ['job', 'guard', 'guard_abort', 'data', 'parameters', 'batch', 'epoch_gap',
    'updates', 'shortened', 'patience', 'nonfinite', 'primary', 'paper_claim', 'TAB_claim'])
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
    elif fault == 'shortened':
        r['epoch_history'].pop(); r['actual_completed_epochs'] = 999; r['optimizer_updates'] = 9990
    elif fault == 'patience': r['early_stopping_patience_checks'] = 3
    elif fault == 'nonfinite': r['test_metrics']['test/other'] = float('nan')
    elif fault == 'primary': r['primary_heldout_W1'] = .9
    elif fault == 'paper_claim': r['historical_paper_equivalence_certified'] = True
    else: r['strict_TAB_result'] = True
    with pytest.raises(ValueError): validate_record(q, j, r, g)
