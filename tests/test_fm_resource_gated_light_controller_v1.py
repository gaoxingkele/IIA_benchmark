from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from scripts.flow_matching.run_resource_gated_light_controller_v1 import GuardedChild, guarded_factory, resource_api


def test_full_child_exit_releases_slot_and_preserves_command(tmp_path):
    released = []
    @contextmanager
    def slot():
        try:
            yield
        finally:
            released.append(True)
    reservation = slot(); reservation.__enter__()
    command = [sys.executable, '-c', 'print(sum(range(1000)))']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    child = GuardedChild(process, reservation,
        {'resource_snapshot': lambda: {'commit_available_bytes': 10**9}},
        {'emergency_commit_headroom_bytes': 0, 'guard_poll_seconds': .001, 'controller_yield_seconds': 0},
        lambda _: None, tmp_path / 'receipt.json', {'command': command})
    assert child.wait(timeout=10) == 0
    assert process.stdout.read().strip() == b'499500'
    assert released == [True]
    assert child.poll() == 0
    assert released == [True]
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['command'] == command
    assert receipt['exit_code'] == 0
    assert receipt['benchmark_result'] is False


def test_observation_timeout_does_not_kill_or_release_live_child(tmp_path):
    released = []
    @contextmanager
    def slot():
        try:
            yield
        finally:
            released.append(True)
    reservation = slot(); reservation.__enter__()
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.15)'])
    child = GuardedChild(process, reservation,
        {'resource_snapshot': lambda: {'commit_available_bytes': 10**9}},
        {'emergency_commit_headroom_bytes': 0, 'guard_poll_seconds': .001, 'controller_yield_seconds': 0},
        lambda _: None, tmp_path / 'receipt.json', {})
    with pytest.raises(subprocess.TimeoutExpired):
        child.wait(timeout=.01)
    assert process.poll() is None
    assert released == []
    assert child.wait(timeout=10) == 0
    assert released == [True]


def test_shared_slot_serializes_two_unchanged_registered_commands(tmp_path):
    queue_path = tmp_path / 'queue.json'
    queue = {'jobs': [{'id': 'first'}, {'id': 'second'}]}
    queue_path.write_text(json.dumps(queue))
    settings = dict(minimum_available_memory_bytes=0, minimum_commit_headroom_bytes=0,
        emergency_commit_headroom_bytes=0, guard_poll_seconds=.001, controller_yield_seconds=0)
    waiting = threading.Event()
    def update(state):
        if state.get('requested_job') == 'second':
            waiting.set()
    factory = guarded_factory(resource_api(), settings, tmp_path / 'shared.lock',
        tmp_path / 'state', update, queue_path, queue)
    def command(job):
        return [sys.executable, '-c', 'print(12345)', '--queue', str(queue_path), '--job-id', job]
    first = factory(command('first'), stdout=subprocess.PIPE)
    second, errors = [], []
    def launch_second():
        try:
            second.append(factory(command('second'), stdout=subprocess.PIPE))
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=launch_second)
    worker.start()
    assert waiting.wait(10)
    assert second == []
    assert first.wait(timeout=10) == 0
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert errors == []
    assert len(second) == 1
    assert second[0].wait(timeout=10) == 0
    for job, child in [('first', first), ('second', second[0])]:
        assert child.stdout.read().strip() == b'12345'
        receipt = json.loads((tmp_path / 'state' / 'receipts' / (job + '.json')).read_text())
        assert receipt['command'] == command(job)
