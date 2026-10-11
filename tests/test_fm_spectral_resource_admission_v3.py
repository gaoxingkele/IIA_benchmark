import ast
import copy
import inspect

import pytest

from scripts.flow_matching import run_spectral_original_queue as original
from scripts.flow_matching.run_spectral_resource_admission_v3 import admission_settings, selected_main_ast
from scripts.flow_matching.run_spectral_resource_admission_v3 import idle


def test_full_original_main_has_only_one_scheduler_statement_added():
    source = inspect.getsource(original.main)
    actual = selected_main_ast(source)
    dispatch = next(node for node in ast.walk(actual) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'job')
    added = dispatch.body.pop(0)
    assert ast.dump(added) == ast.dump(ast.parse(
        "queue['settings'] = admission_settings(queue, job, registered_settings)").body[0])
    assert ast.dump(actual) == ast.dump(ast.parse(source))


def test_full_spectral_only_admission_increases_without_mutating_other_guards():
    settings = dict(minimum_commit_headroom_bytes=20 * 1024**3,
        emergency_commit_headroom_bytes=12 * 1024**3, emergency_gpu_free_bytes=4 * 1024**3,
        minimum_available_memory_bytes=12 * 1024**3)
    snapshot = copy.deepcopy(settings)
    queue = dict(spectral_sampling_minimum_commit_headroom_bytes=48 * 1024**3)
    spectral = admission_settings(queue, {'model': 'spectral_flow'}, settings)
    assert spectral['minimum_commit_headroom_bytes'] == 48 * 1024**3
    spectral['minimum_commit_headroom_bytes'] = settings['minimum_commit_headroom_bytes']
    assert spectral == settings == snapshot
    assert admission_settings(queue, {'model': 'diffusion_ts'}, settings) == snapshot
    queue['spectral_sampling_minimum_commit_headroom_bytes'] = 1
    with pytest.raises(ValueError):
        admission_settings(queue, {'model': 'spectral_flow'}, settings)


@pytest.mark.parametrize('source', ['def main():\n    pass\n',
    "def main():\n    for job in queue['jobs']: pass\n    for job in queue['jobs']: pass\n"])
def test_ambiguous_original_dispatch_is_rejected(source):
    with pytest.raises(ValueError):
        selected_main_ast(source)


@pytest.mark.parametrize('fault', ['pid', 'creation_time', 'command', 'active_pid',
    'active_job', 'owned_lock', 'running_state', 'native_child'])
def test_idle_handoff_never_accepts_an_active_or_reused_worker(fault, monkeypatch):
    expected = dict(pid=7, create_time=12.0, command=['full-controller'])
    state = dict(pid=7, state='waiting_for_heavy_lock', active_pid=None,
                 active_job=None, shared_lock_owned=False)
    class Process:
        pid = 7
        ct = 12.0
        command = ['full-controller']
        descendants = []
        def create_time(self): return self.ct
        def cmdline(self): return self.command
        def children(self, recursive): return self.descendants
    parent = Process()
    monkeypatch.setattr('scripts.flow_matching.run_spectral_resource_admission_v3.console_helper', lambda p: False)
    assert idle(parent, state, expected)
    if fault == 'pid': parent.pid = 8
    elif fault == 'creation_time': parent.ct = 13.0
    elif fault == 'command': parent.command = ['other-controller']
    elif fault == 'active_pid': state['active_pid'] = 9
    elif fault == 'active_job': state['active_job'] = 'full-job'
    elif fault == 'owned_lock': state['shared_lock_owned'] = True
    elif fault == 'running_state': state['state'] = 'running'
    else: parent.descendants = [object()]
    assert not idle(parent, state, expected)
