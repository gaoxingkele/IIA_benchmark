from types import SimpleNamespace
import hashlib
import json

import pytest

from scripts.flow_matching import audit_post_resource_incident_execution_v1 as audit


@pytest.mark.parametrize('pid', [None, False, True, 0, -1, '123'])
def test_idle_worker_never_attests_to_current_auditor(monkeypatch, pid):
    def forbidden(_):
        pytest.fail('An invalid worker identity must not call psutil.Process')
    monkeypatch.setattr(audit.psutil, 'Process', forbidden)
    assert audit.process_identity(pid) == dict(live=False, reason='no_positive_integer_pid')


def test_reused_pid_cannot_certify_original_worker(monkeypatch):
    monkeypatch.setattr(audit.psutil, 'Process', lambda _: SimpleNamespace(
        create_time=lambda: 20.0, cmdline=lambda: ['python', 'original_worker']))
    evidence = audit.process_identity(123, 10.0, 'original_worker')
    assert evidence['live'] is False and evidence['reason'] == 'pid_reused'


def test_unrelated_process_cannot_certify_registered_controller(monkeypatch):
    monkeypatch.setattr(audit.psutil, 'Process', lambda _: SimpleNamespace(
        create_time=lambda: 10.0, cmdline=lambda: ['python', 'other_worker']))
    evidence = audit.process_identity(123, 10.0, 'registered_controller')
    assert evidence['live'] is False and evidence['reason'] == 'command_identity_differs'


def test_exact_live_identity_records_cpu_and_command(monkeypatch):
    monkeypatch.setattr(audit.psutil, 'Process', lambda _: SimpleNamespace(
        create_time=lambda: 10.0, cmdline=lambda: ['python', 'registered_controller'],
        is_running=lambda: True, cpu_times=lambda: (2.0, 3.0)))
    evidence = audit.process_identity(123, 10.0, 'registered_controller')
    assert evidence['live'] is True and evidence['create_time_verified'] is True
    assert evidence['cpu_seconds'] == 5.0


def test_missing_process_is_terminal_observation_not_completion(monkeypatch):
    def missing(pid):
        raise audit.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(audit.psutil, 'Process', missing)
    assert audit.process_identity(123)['reason'] == 'process_missing'


def test_live_state_checksum_binds_the_exact_parsed_bytes(monkeypatch, tmp_path):
    content = json.dumps(dict(pid=None,active_pid=None,state='waiting')).encode('utf-8')
    (tmp_path/'state.json').write_bytes(content)
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    def forbidden(_):
        pytest.fail('A separate live-state checksum read can bind a newer state')
    monkeypatch.setattr(audit,'sha',forbidden)
    record=audit.capture_state('state.json','worker')
    assert record['sha256_at_capture']==hashlib.sha256(content).hexdigest()
    assert record['state']['state']=='waiting'
    assert record['controller_identity']['live'] is False
