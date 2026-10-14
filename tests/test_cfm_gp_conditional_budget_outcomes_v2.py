"""Protect full five-seed accounting and missing metrics for actual numerical failures."""
import copy
import pytest
from scripts.flow_matching.audit_cfm_gp_conditional_budget_outcomes_v2 import summarize_outcomes


def outcomes(failed=True):
    rows = []
    for seed in range(1103, 1108):
        row = dict(seed=seed)
        for epoch in ('400', '600'):
            failure = failed and seed == 1104 and epoch == '600'
            row[epoch] = dict(status='full_once_observed_numerical_failure' if failure else 'complete_finite_full_grid',
                MSE=None if failure else (seed - 1102) / 1000,
                failed_trajectory_ids=[284] if failure else [])
        rows.append(row)
    return rows


def test_failed_seed_retained_and_full_mean_undefined():
    result = summarize_outcomes(outcomes())
    assert result['400']['finite_full_grid_models'] == 5
    assert result['400']['full_five_seed_MSE_mean'] == pytest.approx(.003)
    assert result['600']['full_once_observed_failures'] == 1
    assert result['600']['observed_failure_fraction'] == .2
    assert result['600']['full_five_seed_MSE_mean'] is None
    assert result['600']['full_five_seed_MSE_sample_std'] is None
    assert not result['600']['successful_subset_mean_published']


def test_all_finite_full_mean_available_only_with_all_five():
    result = summarize_outcomes(outcomes(False))
    assert result['600']['full_five_seed_MSE_mean'] == pytest.approx(.003)
    assert result['600']['full_five_seed_MSE_sample_std'] > 0


@pytest.mark.parametrize('mutation', ['drop_failure', 'duplicate_seed', 'unknown_seed', 'pending', 'subset_metric', 'no_failed_id'])
def test_invalid_completion_rejected(mutation):
    rows = copy.deepcopy(outcomes())
    if mutation == 'drop_failure': rows.pop(1)
    elif mutation == 'duplicate_seed': rows[1]['seed'] = 1103
    elif mutation == 'unknown_seed': rows[1]['seed'] = 9999
    elif mutation == 'pending': rows[1]['600']['status'] = 'pending'
    elif mutation == 'subset_metric': rows[1]['600']['MSE'] = .001
    elif mutation == 'no_failed_id': rows[1]['600']['failed_trajectory_ids'] = []
    with pytest.raises(ValueError): summarize_outcomes(rows)


@pytest.mark.parametrize('metric', [float('nan'), float('inf'), -.001, None])
def test_success_needs_real_finite_nonnegative_metric(metric):
    rows = outcomes(False)
    rows[1]['600']['MSE'] = metric
    with pytest.raises(ValueError): summarize_outcomes(rows)
