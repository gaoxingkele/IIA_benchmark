import pytest
from scripts.flow_matching.giflow_update_budget_v2 import validate_observed_updates


def test_original_drop_last_retained_instead_of_inventing_tail_updates():
    loader=dict(samples=6267,batch_size=128,batches=48,drop_last=True)
    assert validate_observed_updates(300,300,14400,loader,{str(e):48 for e in range(300)}) == dict(updates_per_epoch=48,dropped_tail_per_epoch=123)
    with pytest.raises(ValueError):
        validate_observed_updates(300,300,14700,loader,{str(e):49 for e in range(300)})


def test_non_dropped_tail_is_counted_and_missing_epoch_rejected():
    loader=dict(samples=6267,batch_size=128,batches=49,drop_last=False)
    validate_observed_updates(300,300,14700,loader,{str(e):49 for e in range(300)})
    with pytest.raises(ValueError):
        validate_observed_updates(300,300,14700,loader,{str(e):49 for e in range(299)})
