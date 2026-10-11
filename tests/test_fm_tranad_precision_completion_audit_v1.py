import numpy as np
import pytest

from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays


@pytest.fixture
def archives(tmp_path):
    source = dict(e000_train=np.ones((4, 2)), e000_validation=np.ones((3, 2)),
                  e000_test=np.ones((5, 2)), e000_labels=np.array([0, 1, 1, 0, 0]))
    scores = {key: np.arange(len(value), dtype=float) if not key.endswith('labels') else value.copy()
              for key, value in source.items()}
    np.savez(tmp_path / 'inputs.npz', **source)
    dataset = dict(entities=[dict(entity_id='entity', array_prefix='e000', test_points=5)])
    return tmp_path, source, scores, dataset


def test_full_original_array_coverage(archives):
    path, _, scores, dataset = archives
    np.savez(path / 'scores.npz', **scores)
    with np.load(path / 'inputs.npz') as inputs, np.load(path / 'scores.npz') as saved:
        entities, lengths, validation = check_arrays(inputs, saved, dataset)
    assert lengths == {'entity': 5}
    assert len(entities['entity'][0]) == 5
    np.testing.assert_array_equal(validation, [0, 1, 2])


@pytest.mark.parametrize('defect', ['labels', 'nonfinite', 'dropped_tail', 'extra_key'])
def test_reject_incomplete_or_changed_diagnostic(archives, defect):
    path, _, scores, dataset = archives
    if defect == 'labels':
        scores['e000_labels'][0] = 1
    elif defect == 'nonfinite':
        scores['e000_validation'][0] = np.inf
    elif defect == 'dropped_tail':
        scores['e000_test'] = scores['e000_test'][:-1]
    else:
        scores['extra_entity_test'] = np.zeros(1)
    np.savez(path / 'scores.npz', **scores)
    with np.load(path / 'inputs.npz') as inputs, np.load(path / 'scores.npz') as saved:
        with pytest.raises(ValueError):
            check_arrays(inputs, saved, dataset)
