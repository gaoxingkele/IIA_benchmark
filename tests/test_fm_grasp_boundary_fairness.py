import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import json
import sys
import pytest

from scripts.flow_matching.run_grasp_fair_queue_v4 import bound_main
from scripts.flow_matching import run_grasp_protocol_queue as original
from scripts.flow_matching import transition_grasp_at_job_boundary as handoff


def test_only_post_slot_yield_and_original_cuda_default_binding_change():
    source = Path(original.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    function = bound_main()
    assert function.__globals__['guarded_gpu_wait'] is original.guarded_gpu_wait
    assert function.__globals__['complete'] is original.complete
    assert function.__globals__['verify'] is original.verify
    # Count the injected sleep outside resource ownership, without dispatching.
    import dis
    calls = list(dis.get_instructions(function))
    assert sum(i.opname == 'LOAD_ATTR' and i.argval == 'sleep' for i in calls) == 2
    assert 'scripts.flow_matching.grasp_protocol_cuda_runtime' in function.__code__.co_consts
    assert 'configs/experiments/fm_grasp_protocol_queue.v4.json' in function.__code__.co_consts


def test_real_controller_releases_slot_before_configured_yield(tmp_path, monkeypatch):
    function = bound_main(); namespace = function.__globals__
    events = []
    @contextmanager
    def slot(*args):
        events.append('slot_owned')
        yield {}
        events.append('slot_released')
    api = {'exclusive_lock': lambda path: object(), 'resource_slot': slot,
           'wait_for_headroom': lambda *args: {}}
    settings = {'minimum_gpu_free_bytes': 1, 'cpu_threads': 2, 'gpu_index': 0}
    queue = {'state_root': 'run', 'runtime_config': 'runtime.json', 'resource_lock': 'gpu.lock',
             'settings': settings, 'python': sys.executable, 'controller_yield_seconds': 2,
             'jobs': [{'id': 'full_job', 'output_directory': 'output'}]}
    (tmp_path / 'queue.json').write_text(json.dumps(queue))
    (tmp_path / 'runtime.json').write_text('{}')
    namespace.update(ROOT=tmp_path, verify=lambda queue: None,
        load_controller=lambda *args: (None, api, None), gpu_snapshot=lambda settings: {'gpu_free_bytes': 10},
        complete=lambda job: 'guard_called' in events,
        guarded_gpu_wait=lambda child, supplied, api, update: (events.append('guard_called') or 0, {}),
        time=SimpleNamespace(sleep=lambda seconds: events.append(('sleep', seconds))))
    monkeypatch.setattr(namespace['subprocess'], 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=123))
    monkeypatch.setattr(sys, 'argv', ['controller', '--queue', str(tmp_path / 'queue.json')])
    function()
    assert events == ['slot_owned', 'guard_called', 'slot_released', ('sleep', 2)]


@pytest.mark.parametrize('case', ['ready', 'live_model', 'new_model', 'partial_result', 'reused_parent', 'wrong_module', 'wrong_queue'])
def test_only_terminal_completed_childless_boundary_is_eligible(monkeypatch, case):
    expected = {'pid': 42, 'create_time': 123.0, 'module': 'original.native.module', 'job_ids': {'full_model'},
                'queue_path': str(Path('queue.json').resolve())}
    state = {'active_job': 'full_model', 'active_pid': 100}
    parent = SimpleNamespace(pid=42,
        create_time=lambda: 124.0 if case == 'reused_parent' else 123.0,
        cmdline=lambda: ['other.module'] if case == 'wrong_module' else ['original.native.module', '--queue',
                        str(Path('other.json' if case == 'wrong_queue' else 'queue.json').resolve())],
        children=lambda recursive=True: [object()] if case == 'new_model' else [])
    monkeypatch.setattr(handoff.psutil, 'pid_exists', lambda pid: case == 'live_model')
    monkeypatch.setattr(handoff, 'console_helper', lambda child: False)
    observed = handoff.boundary_observation(parent, state, expected, lambda job: case != 'partial_result')
    assert (observed == 'full_model') == (case == 'ready')
