from types import SimpleNamespace

import pytest

from scripts.flow_matching import heavy_scheduler_v2 as scheduler
from scripts.flow_matching.heavy_resources import waited_lock


SETTINGS = {'minimum_available_memory_bytes': 24, 'minimum_commit_headroom_bytes': 16}
GOOD = {'physical_available_bytes': 24, 'commit_available_bytes': 16}
LOW = {'physical_available_bytes': 24, 'commit_available_bytes': 15}


class Handle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_insufficient_memory_never_acquires_shared_lock(monkeypatch, tmp_path):
    snapshots = iter([LOW, GOOD, GOOD])
    handle = Handle()
    acquired = []
    observed = []
    monkeypatch.setattr(scheduler, 'resource_snapshot', lambda: next(snapshots))
    monkeypatch.setattr(scheduler, 'exclusive_lock', lambda p: acquired.append(p) or handle)
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    with scheduler.resource_slot(tmp_path / 'lock', SETTINGS, observed.append) as snapshot:
        assert snapshot == GOOD and not handle.closed
        assert len(acquired) == 1
        assert observed == [dict(LOW, state='waiting_for_memory', shared_lock_owned=False)]
    assert handle.closed


def test_resource_drop_after_acquisition_releases_before_wait(monkeypatch, tmp_path):
    snapshots = iter([GOOD, LOW, GOOD, GOOD])
    first, second = Handle(), Handle()
    handles = iter([first, second])
    monkeypatch.setattr(scheduler, 'resource_snapshot', lambda: next(snapshots))
    monkeypatch.setattr(scheduler, 'exclusive_lock', lambda p: next(handles))
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: assert_closed(first))
    with scheduler.resource_slot(tmp_path / 'lock', SETTINGS):
        assert first.closed and not second.closed
    assert second.closed


def assert_closed(handle):
    assert handle.closed


def test_post_acquisition_error_releases_lock(monkeypatch, tmp_path):
    handle = Handle()
    calls = iter([GOOD, RuntimeError('probe failed')])
    def probe():
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(scheduler, 'resource_snapshot', probe)
    monkeypatch.setattr(scheduler, 'exclusive_lock', lambda p: handle)
    with pytest.raises(RuntimeError, match='probe failed'):
        with scheduler.resource_slot(tmp_path / 'lock', SETTINGS):
            pass
    assert handle.closed


def test_binding_preserves_lane_lock_and_requires_reservation(monkeypatch, tmp_path):
    handle = Handle()
    controller = SimpleNamespace(waited_lock=waited_lock)
    scheduler.bind_controller(controller, tmp_path, SETTINGS)
    with pytest.raises(RuntimeError, match='owned shared slot'):
        controller.wait_for_headroom(SETTINGS)
    monkeypatch.setattr(scheduler, 'resource_snapshot', lambda: GOOD)
    monkeypatch.setattr(scheduler, 'exclusive_lock', lambda p: handle)
    shared = tmp_path / 'experiments/runs/fm_heavy_recovery_cpu.lock'
    with controller.waited_lock(shared):
        assert controller.wait_for_headroom(SETTINGS) == GOOD
        with pytest.raises(RuntimeError):
            controller.wait_for_headroom(dict(SETTINGS, minimum_commit_headroom_bytes=8))
    assert handle.closed
    # A genuine lane lock still acquires a Windows/POSIX advisory file lock.
    with controller.waited_lock(tmp_path / 'lane.lock') as lane:
        assert lane is not None


def test_busy_lock_wait_does_not_claim_ownership(monkeypatch, tmp_path):
    handle = Handle()
    handles = iter([None, handle])
    states = []
    monkeypatch.setattr(scheduler, 'resource_snapshot', lambda: GOOD)
    monkeypatch.setattr(scheduler, 'exclusive_lock', lambda p: next(handles))
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    with scheduler.resource_slot(tmp_path / 'lock', SETTINGS, states.append):
        assert states == [dict(GOOD, state='waiting_for_heavy_lock', shared_lock_owned=False)]
    assert handle.closed


def test_registered_successors_preserve_all_jobs_and_limits():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    read = lambda name: json.loads((root / 'configs/experiments' / name).read_text(encoding='utf-8'))
    old, new = read('fm_crossad_complete_queue.v5.json'), read('fm_crossad_complete_queue.v6.json')
    assert len(new['jobs']) == len(old['jobs']) == 139
    for a, b in zip(old['jobs'], new['jobs']):
        assert {k: v for k, v in a.items() if k != 'frozen_files'} == {k: v for k, v in b.items() if k != 'frozen_files'}
        assert b['frozen_files'][:len(a['frozen_files'])] == a['frozen_files']
    for key in ('minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'):
        assert old['settings'][key] == new['settings'][key]
    assert old['settings']['preflight_parent_sha256'] == new['settings']['preflight_parent_sha256']
    old, new = read('fm_resource_recovery_queue.v2.json'), read('fm_resource_recovery_queue.v3.json')
    assert len(new['jobs']) == 11 and new['jobs'] == old['jobs']
    for key in ('minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'):
        assert old['settings'][key] == new['settings'][key]


def test_transition_waits_for_running_experiment_without_suspending(monkeypatch, tmp_path):
    import json
    from scripts.flow_matching import transition_heavy_scheduler_v2 as transition
    state = tmp_path / 'status.json'
    state.write_text(json.dumps({'pid': 123, 'active_pid': 456, 'state': 'running'}))
    process = SimpleNamespace(pid=123, cmdline=lambda: ['python', 'expected_controller'],
                              suspend=lambda: pytest.fail('Active controller must not be suspended'))
    monkeypatch.setattr(transition.psutil, 'Process', lambda pid: process)
    with pytest.raises(transition.BusyController, match='Live experiment'):
        transition.stop_idle(state, 'expected_controller', tmp_path / 'archive.json')
    assert not (tmp_path / 'archive.json').exists()


def test_transition_resumes_if_child_dispatched_before_suspension(monkeypatch, tmp_path):
    import json
    from scripts.flow_matching import transition_heavy_scheduler_v2 as transition
    state = tmp_path / 'status.json'
    state.write_text(json.dumps({'pid': 123, 'active_pid': None, 'state': 'waiting_for_memory'}))
    calls = []
    child = SimpleNamespace(cmdline=lambda: ['python', 'real_experiment.py'])
    process = SimpleNamespace(cmdline=lambda: ['python', 'expected_controller'], suspend=lambda: calls.append('suspended'),
                              resume=lambda: calls.append('resumed'), is_running=lambda: True,
                              children=lambda recursive: [child])
    monkeypatch.setattr(transition.psutil, 'Process', lambda pid: process)
    with pytest.raises(transition.BusyController, match='Experiment child exists'):
        transition.stop_idle(state, 'expected_controller', tmp_path / 'archive.json')
    assert calls == ['suspended', 'resumed']
    assert not (tmp_path / 'archive.json').exists()
