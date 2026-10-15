import pytest

from scripts.flow_matching import run_pi_causal1107_deferred_full_replay_v2 as sut


@pytest.mark.parametrize('error', [RuntimeError('worker exited'), sut.psutil.NoSuchProcess(123)])
def test_source_finishing_between_read_and_pid_query_uses_complete_artifacts(monkeypatch, error):
    def raced(config, queue):raise error
    monkeypatch.setattr(sut.original, 'ready_or_exact_worker_live', raced)
    monkeypatch.setattr(sut.original.registration, 'require_original_complete', lambda queue: ('job','complete'))
    assert sut.ready_or_exact_worker_live({}, {}) is True


def test_absent_worker_and_incomplete_source_preserve_failure_without_retry(monkeypatch):
    error=RuntimeError('Original worker absent without verified full result')
    def absent(config, queue):raise error
    def incomplete(queue):raise FileNotFoundError('result')
    monkeypatch.setattr(sut.original, 'ready_or_exact_worker_live', absent)
    monkeypatch.setattr(sut.original.registration, 'require_original_complete', incomplete)
    with pytest.raises(RuntimeError) as result:sut.ready_or_exact_worker_live({}, {})
    assert result.value is error


def test_pending_live_source_is_not_promoted(monkeypatch):
    monkeypatch.setattr(sut.original, 'ready_or_exact_worker_live', lambda config, queue:False)
    def no_promotion(queue):raise AssertionError('Do not bypass the original pending source gate')
    monkeypatch.setattr(sut.original.registration, 'require_original_complete', no_promotion)
    assert sut.ready_or_exact_worker_live({}, {}) is False


def test_only_idle_canonical_metadata_watcher_can_be_replaced():
    state=dict(pid=1,create_time=2.,state='waiting_exact_original_full_training_result',
               canonical_id=sut.original.replay.CANONICAL,active_pid=None)
    assert sut.idle_watcher(state,dict(pid=1,create_time=2.),[])
    assert not sut.idle_watcher(dict(state,active_pid=3),dict(pid=1,create_time=2.),[])
    assert not sut.idle_watcher(dict(state,state='running_original_point_native_independent_replay'),dict(pid=1,create_time=2.),[])
