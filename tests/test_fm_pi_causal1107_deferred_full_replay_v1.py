import pytest

from scripts.flow_matching import run_pi_causal1107_deferred_full_replay_v1 as sut


def config():
    return dict(source_worker=dict(pid=123, create_time=456., command=['original', '--job-id', sut.replay.CANONICAL]))


def pending(monkeypatch):
    def incomplete(queue):raise FileNotFoundError('Original full result is still pending')
    monkeypatch.setattr(sut.registration, 'require_original_complete', incomplete)


def test_live_pending_worker_waits_without_source_promotion(monkeypatch):
    pending(monkeypatch)
    class Worker:
        def create_time(self):return 456.
        def cmdline(self):return config()['source_worker']['command']
    monkeypatch.setattr(sut.psutil, 'Process', lambda pid: Worker())
    assert sut.ready_or_exact_worker_live(config(), {}) is False


@pytest.mark.parametrize('change', ['ctime', 'command', 'missing'])
def test_missing_or_reused_source_never_causes_training_retry(monkeypatch, change):
    pending(monkeypatch)
    class Worker:
        def create_time(self):return 457. if change=='ctime' else 456.
        def cmdline(self):return ['different'] if change=='command' else config()['source_worker']['command']
    def process(pid):
        if change=='missing':raise sut.psutil.NoSuchProcess(pid)
        return Worker()
    monkeypatch.setattr(sut.psutil, 'Process', process)
    with pytest.raises(RuntimeError):sut.ready_or_exact_worker_live(config(), {})


def test_verified_complete_source_does_not_require_a_still_live_pid(monkeypatch):
    monkeypatch.setattr(sut.registration, 'require_original_complete', lambda queue: ('job', 'complete'))
    def no_pid(pid):raise AssertionError('A fully verified terminal source does not require its retired PID')
    monkeypatch.setattr(sut.psutil, 'Process', no_pid)
    assert sut.ready_or_exact_worker_live(config(), {}) is True


def test_future_replay_uses_same_cached_host_bootstrap_and_exact_runtime():
    command=sut.replay_command('original-python')
    assert command[:6]==['original-python','-B','-X','utf8','-u','-c']
    assert sut.RUNTIME in command[-1] and sut.REPLAY_MODULE in command[-1]
    assert 'install_cached_loader' in command[-1]
    assert '--worker' not in command and '--job-id' not in command


@pytest.mark.parametrize('change', ['queue', 'interpreter'])
def test_future_registration_cannot_change_original_queue_or_interpreter(monkeypatch, change):
    original = dict(canonical_id=sut.replay.CANONICAL, original_scope=sut.replay.SCOPE,
                    original_budget=sut.replay.BUDGET, completed_replay_runtime=sut.RUNTIME,
                    original_queue=sut.ORIGINAL_QUEUE, poll_seconds=5, python='original-python')
    monkeypatch.setattr(sut.registration, 'read', lambda path: dict(python='original-python'))
    if change == 'queue':original['original_queue']='different-original-queue'
    else:original['python']='different-python'
    with pytest.raises(ValueError):sut.verify(original)
