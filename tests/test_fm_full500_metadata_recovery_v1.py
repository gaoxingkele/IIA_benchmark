import json
from pathlib import Path

import psutil
import pytest

from scripts.flow_matching import resume_anomaly_transformer_full500_metadata_v1 as recovery


def denied(code=5):
    error = PermissionError('fixture Windows file sharing/access conflict')
    error.winerror = code
    return error


def test_status_rename_retries_identical_bytes_without_recomputing(tmp_path):
    target = tmp_path / 'status.json'
    target.write_text('old', encoding='utf-8')
    observed, pauses = [], []
    def replace(source, destination):
        observed.append((Path(source).read_bytes(), Path(destination).read_bytes()))
        if len(observed) < 3:
            raise denied()
        Path(source).replace(destination)
    writer = recovery.status_writer(target, lambda *_: pytest.fail('wrong fallback'),
        replace=replace, pause=pauses.append)
    writer(target, dict(active_pid=None, state='waiting'))
    assert len(observed) == 3 and len(pauses) == 2
    assert len({source for source, _ in observed}) == 1
    assert all(old == b'old' for _, old in observed)
    assert json.loads(target.read_text()) == dict(active_pid=None, state='waiting')
    assert not list(tmp_path.glob('*.metadata-next'))


@pytest.mark.parametrize('code', [5, 32, 33])
def test_terminal_metadata_error_preserves_prior_state_and_diagnostic_temp(tmp_path, code):
    target = tmp_path / 'status.json'; target.write_text('old', encoding='utf-8')
    calls, pauses = [], []
    def replace(*args):
        calls.append(args)
        raise denied(code)
    writer = recovery.status_writer(target, lambda *_: None, replace=replace,
        pause=pauses.append, attempts=3)
    with pytest.raises(PermissionError): writer(target, dict(state='waiting'))
    assert len(calls) == 3 and len(pauses) == 2 and target.read_text() == 'old'
    assert len(list(tmp_path.glob('*.metadata-next'))) == 1


@pytest.mark.parametrize('error', [OSError('disk failure'), denied(87)])
def test_other_errors_are_not_retried(tmp_path, error):
    calls = []
    def replace(*args):
        calls.append(args); raise error
    writer = recovery.status_writer(tmp_path/'status.json', lambda *_: None,
        replace=replace, pause=lambda _: pytest.fail('unexpected retry'))
    with pytest.raises(OSError): writer(tmp_path/'status.json', {})
    assert len(calls) == 1


def test_numerical_proof_write_is_delegated_without_retry(tmp_path):
    calls = []
    def fallback(path, value):
        calls.append((path, value)); raise denied()
    writer = recovery.status_writer(tmp_path/'status.json', fallback,
        replace=lambda *_: pytest.fail('intercepted numerical proof'))
    with pytest.raises(PermissionError): writer(tmp_path/'native_replay.json', {'passed': True})
    assert len(calls) == 1 and not list(tmp_path.glob('*.metadata-next'))


def test_owner_identity_and_missing_pid_are_checked():
    class Process:
        def create_time(self): return 2.
    assert not recovery.owner_missing(dict(pid=123, create_time=2.), lambda _: Process())
    assert recovery.owner_missing(dict(pid=123, create_time=1.), lambda _: Process())
    def missing(pid): raise psutil.NoSuchProcess(pid)
    assert recovery.owner_missing(dict(pid=123, create_time=1.), missing)


def test_recovery_refuses_attempts_or_active_children(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'owner_missing', lambda _: True)
    config = dict(prior_owner=dict(pid=123, create_time=1.), prior_launch='launch.json',
        prior_status='status.json', prior_stderr='stderr.log')
    (tmp_path/'launch.json').write_text(json.dumps(config['prior_owner']))
    state = dict(**config['prior_owner'], state='waiting_actual_full_training_completion',
        active_pid=None, verified_model_count=0, failed_preserved={}, required_models=500)
    (tmp_path/'status.json').write_text(json.dumps(state))
    (tmp_path/'stderr.log').write_text('PermissionError: [WinError 5] status.next.json os.replace(temporary, path)')
    original = dict(state_root='state', artifact_root=str(tmp_path/'artifacts'))
    assert recovery.stopped_before_any_numerical_attempt(config, original)['numerical_attempts_found'] == 0
    state['active_pid'] = 999
    (tmp_path/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='idle never-started'): recovery.stopped_before_any_numerical_attempt(config, original)
    state['active_pid'] = None; (tmp_path/'status.json').write_text(json.dumps(state))
    (tmp_path/'state/attempts/job/native').mkdir(parents=True)
    with pytest.raises(ValueError, match='Numerical attempts'): recovery.stopped_before_any_numerical_attempt(config, original)


def test_real_frozen500_contract_and_recovery_pins():
    path = recovery.ROOT/'configs/runtime/fm_anomaly_transformer_full500_metadata_recovery.v1.json'
    original, cases = recovery.verify_recovery(recovery.runner.read(path))
    assert len(cases) == 500
    assert original['formula_tolerance'] == dict(rtol=2e-6, atol=2e-6)
    assert original['native_array_comparison'] == 'exact_dtype_shape_and_content_bytes'
