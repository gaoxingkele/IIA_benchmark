import copy

import numpy as np
import pytest

from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, prediction_metrics, independent_stats


def case():
    truth = np.arange(8, dtype=float).reshape(2, 4, 1)
    predictions = truth + np.array([1., 2.])[:, None, None]
    metrics, per = prediction_metrics(predictions, truth)
    job = dict(epochs=400, expected_optimizer_updates=400, training_items=160,
        test_trajectories=2, test_points_per_trajectory=4, observed_dimensions=1)
    result = dict(epochs_completed=400, optimizer_updates=400, training_items=160,
        training_losses=[1.]*400, metrics=metrics, training_trajectory_ids=[0], test_trajectory_ids=[0, 1])
    data = dict(train_ids=np.array([0]), test_ids=np.array([0, 1]), test_states=truth,
        test_times=np.tile(np.arange(4), (2, 1)), initial_states=np.array([[0.], [4.]]))
    arrays = dict(predictions=predictions, truth=truth.copy(), times=data['test_times'].copy(),
        test_ids=data['test_ids'].copy(), initial_states=data['initial_states'].copy(), per_trajectory_MSE=per)
    return job, result, arrays, data


def test_all_full_metric_types_are_recomputed():
    job, result, arrays, data = case()
    measured = audit_arrays(job, result, arrays, data)
    assert measured['MSE'] == 2.5 and measured['MAE'] == 1.5
    assert measured['RMSE'] == pytest.approx(2.5**.5)
    assert measured['MSE_by_dimension'] == [2.5]


@pytest.mark.parametrize('metric', ['MSE', 'MAE', 'RMSE', 'trajectory_MSE_std', 'trajectory_MSE_mean'])
def test_corrupt_metric_is_not_published(metric):
    job, result, arrays, data = case()
    result['metrics'][metric] += .1
    with pytest.raises(ValueError, match='Independent full metric'):
        audit_arrays(job, result, arrays, data)


def test_partial_budget_wrong_grid_and_wrong_initial_states_are_rejected():
    for field in ['budget', 'grid', 'initial']:
        job, result, arrays, data = case()
        if field == 'budget':
            result['optimizer_updates'] -= 1
        elif field == 'grid':
            arrays['times'][0, 0] = -1
        else:
            arrays['initial_states'][0, 0] = -1
        with pytest.raises(ValueError):
            audit_arrays(job, result, arrays, data)


def test_missing_and_single_seed_do_not_gain_uncertainty():
    assert independent_stats([])['mean'] is None
    assert independent_stats([0.])['mean'] == 0.
    assert independent_stats([1.])['ci95_low'] is None
    assert independent_stats([1., 2., 3., 4., 5.])['n'] == 5
