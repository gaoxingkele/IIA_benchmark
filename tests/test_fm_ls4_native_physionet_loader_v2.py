import pytest

from scripts.flow_matching.audit_ls4_native_physionet_loader_v2 import original_batch_budget


@pytest.mark.parametrize('batch,train_batches,test_batches,updates', [(64, 100, 25, 50000), (32, 200, 50, 100000)])
def test_both_original_yaml_batch_and_epoch_budgets_are_preserved(batch, train_batches, test_batches, updates):
    yaml = dict(data=dict(n=8000, channel=41, classify=False), optim=dict(batch_size=batch, epochs=500))
    assert original_batch_budget(yaml, 6400, 1600) == (batch, train_batches, test_batches)
    assert 500 * train_batches == updates


def test_reduced_patient_or_variable_scope_rejected():
    yaml = dict(data=dict(n=4000, channel=37, classify=False), optim=dict(batch_size=32))
    with pytest.raises(ValueError, match='Full original'):
        original_batch_budget(yaml, 3200, 800)
