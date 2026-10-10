import pytest

from scripts.flow_matching.audit_giflow_checkpoint_replay_resume_v3 import batch_coverage


def test_completed_batch_accumulation_cannot_change_registered_prefix_count():
    prefix=[dict(preserved_completed_batch=True) for _ in range(9)]
    complete=list(prefix)
    complete.extend(dict(batch=i) for i in range(9,14))
    assert len(prefix)==9
    assert batch_coverage(complete,len(prefix),100)==dict(preserved_completed_native_batches=9,newly_replayed_native_batches=5)


def test_reject_alias_corrupted_prefix_count_or_missing_native_forwards():
    complete=[dict(preserved_completed_batch=True) for _ in range(9)]+[dict(batch=i) for i in range(9,14)]
    with pytest.raises(ValueError):batch_coverage(complete,14,100)
    with pytest.raises(ValueError):batch_coverage(complete,9,99)
    complete[9]['preserved_completed_batch']=True
    with pytest.raises(ValueError):batch_coverage(complete,9,100)
