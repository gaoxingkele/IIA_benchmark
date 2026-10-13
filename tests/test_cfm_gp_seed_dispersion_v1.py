import numpy as np
import pytest
import json
from pathlib import Path
from scripts.flow_matching.audit_cfm_gp_seed_dispersion_v1 import dispersion


def inputs(errors):
    values = np.sqrt(np.asarray(errors))[:, None, None]
    n = len(errors)
    return errors, np.arange(n), np.zeros((n, 1)), np.zeros((n, 1)), values, np.zeros_like(values)


def test_large_trajectory_stays_in_full_metric_and_concentration():
    result = dispersion(*inputs([1., 1., 98.]))
    assert result['full_MSE'] == pytest.approx(100. / 3.)
    assert result['top1_squared_error_fraction'] == .98
    assert result['largest_error_trajectories'][0]['trajectory_id'] == 2
    assert result['trajectories_removed'] == 0
    assert result['all_trajectory_MSE'] == [1., 1., 98.]


def test_zero_errors_do_not_produce_nonfinite_fractions():
    result = dispersion(*inputs([0., 0.]))
    assert result['top1_squared_error_fraction'] == result['top5_squared_error_fraction'] == 0.
    assert result['full_MSE'] == 0.


@pytest.mark.parametrize('errors', [[float('nan')], [-1.], [float('inf')]])
def test_nonfinite_or_negative_errors_rejected(errors):
    with np.errstate(invalid='ignore'):
        with pytest.raises(ValueError, match='finite nonnegative'):
            dispersion(*inputs(errors))


def test_worst_coordinate_retains_actual_time_and_signed_values():
    predictions = np.array([[[1.], [-10.]]])
    truth = np.array([[[0.], [2.]]])
    result = dispersion([72.5], [55], np.array([[.3]]), np.array([[.1, .9]]), predictions, truth)
    row = result['largest_error_trajectories'][0]
    assert row['worst_time'] == .9 and row['worst_point_index'] == 1
    assert row['prediction_at_worst_point'] == -10. and row['truth_at_worst_point'] == 2.


def test_actual_frozen_completed_outputs_and_every_seed_route_are_accepted(tmp_path):
    from scripts.flow_matching import audit_cfm_gp_seed_dispersion_v1 as audit
    config = audit.source.read(audit.source.ROOT / 'configs/reproducibility/fm_cfm_gp_sine400_seed_dispersion.2026-10-14_v1.json')
    config['output_report'] = str(tmp_path / 'audit.json')
    audit.run(config)
    report = json.loads(Path(config['output_report']).read_text(encoding='utf-8'))
    assert len(report['records']) == 3
    assert len(report['required_canonical_ids']) == 5
    assert len(report['pending_canonical_ids']) == 2
    assert not report['full_five_seed_training_cohort_complete']
    assert all(r['dispersion']['full_trajectory_count'] == 80 and r['dispersion']['trajectories_removed'] == 0 for r in report['records'])
