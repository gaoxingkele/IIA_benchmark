import pytest

from scripts.flow_matching.run_ls4_native_physionet_v1 import budget, periodic_epochs, ObservedLoader


def test_original_profile_full_budgets():
    base = dict(epochs=500, train_patients=6400, test_patients=1600, variables=41)
    assert budget(dict(base, extrapolation=False, batch_size=64)) == dict(train_batches=100, test_batches=25, optimizer_updates=50000, evaluation_calls=50)
    assert budget(dict(base, extrapolation=True, batch_size=32)) == dict(train_batches=200, test_batches=50, optimizer_updates=100000, evaluation_calls=50)


@pytest.mark.parametrize('field,value', [('epochs',499),('train_patients',6399),('test_patients',1599),('variables',40),('batch_size',64)])
def test_reject_scope_or_extrapolation_batch_reduction(field, value):
    config = dict(epochs=500, train_patients=6400, test_patients=1600, variables=41, extrapolation=True, batch_size=32)
    config[field] = value
    with pytest.raises(ValueError):
        budget(config)


def test_native_zero_indexed_evaluation_schedule():
    epochs = periodic_epochs(500,10)
    assert epochs == list(range(10,500,10))
    assert len(epochs + [-1]) == 50


def test_observer_preserves_identity_and_requires_complete_coverage():
    class Loader:
        batch_size = 2
        def __iter__(self):
            return iter(batches)
        def __len__(self):
            return len(batches)
    batches = [dict(observed_data=[1,2]),dict(observed_data=[3])]
    seen = []
    loader = ObservedLoader(Loader(),3,lambda i,n,d: seen.append(n))
    assert list(loader) == batches
    assert loader.calls == 1 and seen == [2,3]
    assert list(loader)[0] is batches[0]
    with pytest.raises(ValueError):
        list(ObservedLoader(Loader(),4,lambda *a: None))
