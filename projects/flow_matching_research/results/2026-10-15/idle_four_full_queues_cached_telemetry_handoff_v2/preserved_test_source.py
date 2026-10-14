import copy
import json
from types import SimpleNamespace

import pytest

from scripts.flow_matching import run_idle_exact_command_cached_handoff_v2 as sut


OWNER = dict(pid=123, create_time=456.0, command=['python', '-m', 'original'])
IDLE = dict(pid=123, create_time=456.0, state='waiting_for_memory', active_pid=None,
            shared_lock_owned=False)


@pytest.mark.parametrize('change,children', [
    ({'active_pid': 234}, []),
    ({'shared_lock_owned': True}, []),
    ({'pid': 999}, []),
    ({'create_time': 457.0}, []),
    ({'state': 'running'}, []),
    ({}, [dict(name='python.exe', exe='C:/Python/python.exe')]),
    ({}, [dict(name='conhost.exe', exe='C:/other/conhost.exe')]),
])
def test_busy_or_different_owner_cannot_be_stopped(change, children):
    assert not sut.idle_allowed(dict(IDLE, **change), OWNER, children)


def test_legacy_state_uses_measured_os_identity_without_inventing_timestamp():
    state = dict(IDLE)
    del state['create_time']
    assert sut.idle_allowed(state, OWNER, [])
    assert 'create_time' not in state
    assert sut.idle_allowed(IDLE, OWNER, [])


def registration(root):
    roles = {}
    pins = []
    for name, (entry, count) in sut.ENTRIES.items():
        path = root / (name + '.json')
        path.write_text(json.dumps(dict(jobs=[dict(id=str(i)) for i in range(count)])))
        pins.append(dict(path=path.name, sha256=sut.digest(path)))
        arguments = ['--config', name + '.config.json']
        roles[name] = dict(entry_module=entry, original_arguments=arguments,
            previous_owner=dict(command=['python', '-m', entry, *arguments]),
            original_queue=path.name, required_original_slots=count)
    return dict(original_scope=copy.deepcopy(sut.SCOPE), roles=roles, source_receipts=pins)


def test_all_four_full_queues_and_original_arguments_survive(tmp_path):
    config = registration(tmp_path)
    assert [sut.verify(config, name, tmp_path)['required_original_slots']
            for name in sut.ENTRIES] == [70, 9, 175, 25]
    config['roles']['spectral70']['original_arguments'].append('--reduced-budget')
    with pytest.raises(ValueError, match='every original argument'):
        sut.verify(config, 'spectral70', tmp_path)


def test_changed_frozen_bytes_fail_before_handoff(tmp_path):
    config = registration(tmp_path)
    (tmp_path / 'strict175.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen handoff source changed'):
        sut.verify(config, 'strict175', tmp_path)


def test_narrowed_scope_is_rejected(tmp_path):
    config = registration(tmp_path)
    config['original_scope']['method_baseline_ablation_records'] -= 1
    with pytest.raises(ValueError, match='entire original research scope'):
        sut.verify(config, 'strict175', tmp_path)


def test_child_appearing_during_suspension_is_resumed_and_never_stopped(tmp_path, monkeypatch):
    class EndObservation(Exception):
        pass

    class Process:
        suspended = False
        resumed = False
        terminated = False
        calls = 0

        def create_time(self):
            return OWNER['create_time']

        def cmdline(self):
            return OWNER['command']

        def children(self, recursive):
            if self.resumed:
                raise EndObservation()
            if self.suspended:
                return [SimpleNamespace(pid=234, name=lambda: 'python.exe',
                                        exe=lambda: 'C:/Python/python.exe')]
            return []

        def suspend(self):
            self.suspended = True

        def terminate(self):
            self.terminated = True
            raise AssertionError('A numerical child must prevent termination')

        def is_running(self):
            return True

        def resume(self):
            self.resumed = True

    process = Process()
    monkeypatch.setattr(sut.psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(sut, 'read', lambda path: dict(IDLE))
    with pytest.raises(EndObservation):
        sut.handoff(dict(previous_owner=OWNER, previous_state_path='unused'), tmp_path)
    assert process.resumed and not process.terminated
    assert not list(tmp_path.iterdir())
