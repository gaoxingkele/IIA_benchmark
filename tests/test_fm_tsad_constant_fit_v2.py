import numpy as np
import pytest

from scripts.flow_matching import prepare_tsad_execution as old
from scripts.flow_matching import prepare_tsad_constant_fit_v2 as new


RECIPE = dict(train_fraction=.8, purge_points=10, minimum_segment_points=10)


def fixture_arrays():
    index = np.arange(1000, dtype=float)
    train = np.column_stack([np.full(1000, .999), index / 100, .999 + (index % 2) * 1e-14])
    test = np.column_stack([np.linspace(.5, 1, 50), np.arange(50), np.full(50, .999)])
    return train, test, np.zeros(50)


def test_real_constant_roundoff_scale_corrected_all_original_other_channels_retained():
    train, test, labels = fixture_arrays()
    previous, previous_labels, previous_audit = old.split_and_normalize(train, test, labels, RECIPE)
    current, current_labels, audit = new.split_and_normalize(train, test, labels, RECIPE)
    assert previous_audit['std'][0] > 0  # Actual mean/std accumulation roundoff.
    assert audit['std'][0] == 0 and audit['mean'][0] == .999
    assert audit['changed_constant_channels'] == [0]
    for split in previous:
        np.testing.assert_array_equal(previous[split][:, 1:], current[split][:, 1:])
    assert np.all(current['train'][:, 0] == 0)
    assert np.max(np.abs(current['test'][:, 0])) < 1
    assert np.max(np.abs(previous['test'][:, 0])) > 1e12
    np.testing.assert_array_equal(previous_labels, current_labels)
    for key in ('fit_interval', 'purged_interval', 'validation_interval', 'test_points', 'features', 'repairs'):
        assert previous_audit[key] == audit[key]


def test_test_and_validation_never_change_fitted_constant_statistics():
    train, test, labels = fixture_arrays()
    arrays, _, audit = new.split_and_normalize(train, test, labels, RECIPE)
    altered_train = train.copy()
    altered_train[800:, 0] = 1e10
    altered_test = test.copy()
    altered_test[:, 0] = -1e12
    changed, _, another = new.split_and_normalize(altered_train, altered_test, labels, RECIPE)
    for key in ('median', 'mean', 'std', 'constant_channels'):
        assert audit[key] == another[key]
    np.testing.assert_array_equal(arrays['train'], changed['train'])


def test_near_constant_channels_have_no_arbitrary_epsilon_floor():
    train, test, labels = fixture_arrays()
    _, _, audit = new.split_and_normalize(train, test, labels, RECIPE)
    assert audit['std'][2] == audit['legacy_std'][2] > 0
    assert 2 not in audit['constant_channels']


def test_repair_all_constant_nonfinite_fit_before_constant_detection():
    train, test, labels = fixture_arrays()
    train[0, 0] = np.nan
    train[100, 0] = np.inf
    arrays, _, audit = new.split_and_normalize(train, test, labels, RECIPE)
    assert audit['repairs']['train'] == 2
    assert audit['constant_channels'] == [0]
    assert np.all(arrays['train'][:, 0] == 0)


def test_original_complete_preparation_body_unchanged():
    assert new.prepare.__code__ is old.prepare.__code__
    assert new.prepare.__globals__['load_entities'] is old.load_entities
    assert new.prepare.__globals__['split_and_normalize'] is new.split_and_normalize


@pytest.mark.parametrize('shape', [(50, 2), (49,)])
def test_original_label_shape_contract_still_rejected(shape):
    train, test, _ = fixture_arrays()
    with pytest.raises(ValueError):
        new.split_and_normalize(train, test, np.zeros(shape), RECIPE)
