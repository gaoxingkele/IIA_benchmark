import numpy as np
import pytest

from scripts.flow_matching.verify_giflow_corrected_full_replay_v1 import verify_batch


def arrays():
    return dict(prediction=np.array([2., 5.]), target=np.array([1., 4.]),
                masked_flat_positions=np.array([0, 7]), batch_shape=np.array([83, 36, 24, 1]))


def test_retains_native_masked_reductions():
    row = verify_batch(arrays(), arrays(), 83, 2, dict(rtol=.0002, atol=.001))
    assert row['independently_recomputed_metrics']['native_mean_batch_mae'] == 1
    assert row['independently_recomputed_metrics']['native_mean_batch_mse'] == 1
    assert row['independently_recomputed_metrics']['native_mean_batch_mape_percent'] == 62.5


@pytest.mark.parametrize('mutation', ['prediction', 'target', 'positions', 'shape', 'missing', 'nonfinite'])
def test_rejects_incomplete_or_changed_full_replay(mutation):
    original, actual = arrays(), arrays()
    if mutation == 'prediction': actual['prediction'][0] += 1
    elif mutation == 'target': actual['target'][0] += 1
    elif mutation == 'positions': actual['masked_flat_positions'][1] = 0
    elif mutation == 'shape': actual['batch_shape'][0] = 1
    elif mutation == 'missing': actual.pop('target')
    else: actual['prediction'][0] = np.nan
    with pytest.raises(ValueError):
        verify_batch(actual, original, 83, 2, dict(rtol=.0002, atol=.001))
