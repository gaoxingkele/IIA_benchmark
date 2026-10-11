import numpy as np
import pytest

from scripts.flow_matching.verify_brits_electricity_full_replay_v3 import check_arrays


def fixture():
    truth = np.array([[[1., 2.]], [[3., 4.]]], dtype=np.float32)
    mask = np.ones_like(truth)
    arrays = dict(predictions=truth + 1, truth=truth.copy(), mask=mask.copy(),
        test_indices=np.array([0, 1]), unit_ids=np.array([4, 9]),
        group_ids=np.array(['month_a', 'month_b']),
        group_statistics=np.array([[2., 2., 2., 3.], [2., 2., 2., 7.]]))
    return arrays, truth, mask, arrays['unit_ids'], arrays['group_ids']


def test_original_nan_to_zero_loader_truth_is_checked():
    a, t, m, ids, groups = fixture()
    actual, _ = check_arrays(a, t, m, ids, groups, 4)
    assert actual == dict(mae=1., rmse=1., mre=.4)
    t[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='truth, mask'):
        check_arrays(a, t, m, ids, groups, 4)


@pytest.mark.parametrize('fault', ['tail', 'duplicate', 'nonfinite', 'truth', 'mask', 'unit', 'group'])
def test_incomplete_or_mismatched_full_native_replay_is_rejected(fault):
    a, t, m, ids, groups = fixture()
    ids, groups = ids.copy(), groups.copy()
    if fault == 'tail':
        a['predictions'] = a['predictions'][:1]
    elif fault == 'duplicate':
        a['test_indices'] = np.array([0, 0])
    elif fault == 'nonfinite':
        a['predictions'][1, 0, 0] = np.inf
    elif fault == 'truth':
        a['truth'][1, 0, 0] += 1
    elif fault == 'mask':
        a['mask'][1, 0, 0] = 0
    elif fault == 'unit':
        a['unit_ids'][1] = 10
    else:
        a['group_statistics'][1, 0] += 1
    with pytest.raises(ValueError):
        check_arrays(a, t, m, ids, groups, 4)
