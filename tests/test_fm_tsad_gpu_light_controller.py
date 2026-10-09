import dis
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
from types import CodeType, SimpleNamespace

import pytest
from scripts.flow_matching.run_light_controller import ROOT, load_controller
from scripts.flow_matching import transition_tsad_gpu_light_controller as handoff


def config():
    original = json.loads((ROOT / 'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    queue_path = 'configs/experiments/fm_tsad_gpu_queue.v1.json'
    return dict(original, controllers=[{'name': 'tsad_gpu', 'source_path': 'scripts/flow_matching/run_tsad_queue.py',
        'source_functions': ['main'], 'queue_path': queue_path,
        'queue_sha256': hashlib.sha256((ROOT / queue_path).read_bytes()).hexdigest(),
        'status_path': 'experiments/runs/fm_tsad_execution_v1/gpu_status.json',
        'queue_argument': '--queue', 'resource_scheduler_binding': False}])


def instructions(function):
    # A standalone compilation has a different module symbol table.
    return [(i.opname, instructions(i.argval) if isinstance(i.argval, CodeType)
             else i.argval) for i in dis.get_instructions(function)]


def test_original_gpu_controller_and_prerequisite_gate_preserved():
    from scripts.flow_matching import run_tsad_queue, phase_gate
    controller, namespace, _ = load_controller(ROOT, config(), 'tsad_gpu')
    assert instructions(controller.main) == instructions(run_tsad_queue.main)
    assert instructions(namespace['check_queue']) == instructions(phase_gate.check_queue)
    assert instructions(namespace['complete_result']) == instructions(run_tsad_queue.complete_result)
    assert instructions(namespace['exclusive_lock']) == instructions(run_tsad_queue.exclusive_lock)


def test_fresh_tsad_binding_avoids_model_and_data_imports(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(config()), encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'scripts.flow_matching.run_light_controller',
        '--runtime-config', str(path), '--controller', 'tsad_gpu', '--probe-only'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    assert not any(probe[k] for k in ['numpy_imported', 'pandas_imported', 'torch_imported'])


def test_actual_gpu_loop_waits_for_unfinished_prerequisite(tmp_path, monkeypatch):
    runtime = config()
    for source in runtime['source_receipts']:
        destination = tmp_path / source['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source['path'], destination)
    prerequisite = tmp_path / 'prerequisite_config.json'
    prerequisite.write_text(json.dumps({'id': 'unfinished', 'output_root': 'unfinished_output'}))
    (tmp_path / 'prerequisite_queue.json').write_text(json.dumps({'jobs': [{
        'config': prerequisite.name, 'config_sha256': hashlib.sha256(prerequisite.read_bytes()).hexdigest()}]}))
    binding = runtime['controllers'][0]
    queue_path = tmp_path / binding['queue_path']
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({'lane': 'gpu', 'state_root': 'run', 'jobs': [],
                                    'gpu_prerequisite_queues': ['prerequisite_queue.json']}))
    binding['queue_sha256'] = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    controller, namespace, _ = load_controller(tmp_path, runtime, 'tsad_gpu')
    class ObservedWait(Exception): pass
    def wait(seconds): raise ObservedWait()
    def unexpected_child(*args, **kwargs): pytest.fail('Training started before prerequisite completed')
    namespace['time'] = SimpleNamespace(sleep=wait)
    namespace['subprocess'] = SimpleNamespace(Popen=unexpected_child)
    monkeypatch.setattr(namespace['psutil'], 'process_iter', lambda fields: [])
    monkeypatch.setattr(sys, 'argv', ['controller', '--queue', str(queue_path)])
    with pytest.raises(ObservedWait):
        controller.main()
    state = json.loads((tmp_path / 'run/gpu_status.json').read_text())
    assert state['status'] == 'waiting_existing_imputation'
    assert state['pending_prerequisites'] == {'prerequisite_queue.json': 1}


@pytest.mark.parametrize('change', ['none', 'reused_pid', 'active_job', 'child', 'wrong_queue', 'wrong_source', 'wrong_state'])
def test_handoff_requires_exact_observed_idle_identity(tmp_path, monkeypatch, change):
    binding = {'status_path': 'status.json', 'source_path': 'scripts/flow_matching/run_tsad_queue.py',
        'queue_argument': '--queue', 'queue_path': 'queue.json', 'expected_old_pid': 42, 'expected_old_create_time': 123.0}
    state = {'pid': 42, 'status': 'waiting_existing_imputation'}
    if change == 'active_job': state['active_job'] = 'real_training'
    if change == 'wrong_state': state['status'] = 'running'
    (tmp_path / 'status.json').write_text(json.dumps(state), encoding='utf-8')
    command = [sys.executable, str(tmp_path / binding['source_path']), '--queue', str(tmp_path / 'queue.json')]
    if change == 'wrong_queue': command[-1] = str(tmp_path / 'other_queue.json')
    if change == 'wrong_source': command[1] = str(tmp_path / 'other_worker.py')
    process = SimpleNamespace(cmdline=lambda: command,
        children=lambda recursive=True: [object()] if change == 'child' else [],
        create_time=lambda: 124.0 if change == 'reused_pid' else 123.0)
    monkeypatch.setattr(handoff.original, 'ROOT', tmp_path)
    monkeypatch.setattr(handoff.psutil, 'Process', lambda pid: process)
    result = handoff.idle_observation(binding)
    assert (result is not None) == (change == 'none')
