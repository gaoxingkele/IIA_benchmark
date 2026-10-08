import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def fixture_root(tmp_path):
    config = json.loads((ROOT/'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    for record in config['source_receipts']:
        dest = tmp_path/record['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/record['path'], dest)
    for binding in config['controllers']:
        dest = tmp_path/binding['queue_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/binding['queue_path'], dest)
    return config


def test_fresh_probe_does_not_import_data_or_model_libraries():
    result = subprocess.run([sys.executable, '-m', 'scripts.flow_matching.run_light_controller',
        '--runtime-config','configs/runtime/fm_light_controllers.v1.json','--controller','crossad','--probe-only'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert not any(value[k] for k in ('numpy_imported','pandas_imported','torch_imported'))


def test_every_registered_controller_compiles_exact_pinned_sources():
    config = json.loads((ROOT/'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    assert len(config['controllers']) == 7
    for binding in config['controllers']:
        controller, namespace, selected = load_controller(ROOT, config, binding['name'])
        assert selected == binding
        assert controller.main.__code__.co_filename == str(ROOT/binding['source_path'])
        assert namespace['sha'].__code__.co_filename == str(ROOT/'scripts/flow_matching/prepare_tsad_execution.py')
        assert namespace['guarded_wait'].__code__.co_filename == str(ROOT/'scripts/flow_matching/heavy_resources.py')


def test_changed_queue_or_controller_source_is_rejected(tmp_path):
    config = fixture_root(tmp_path)
    source = tmp_path/config['controllers'][0]['source_path']
    source.write_text(source.read_text(encoding='utf-8')+'\n# changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        load_controller(tmp_path, config, 'crossad')
    shutil.copy2(ROOT/config['controllers'][0]['source_path'], source)
    queue = tmp_path/config['controllers'][0]['queue_path']
    queue.write_text('{}')
    with pytest.raises(ValueError, match='queue changed'):
        load_controller(tmp_path, config, 'crossad')


def test_real_original_controller_accepts_complete_receipt_without_running_child(tmp_path, monkeypatch):
    config = fixture_root(tmp_path)
    binding = config['controllers'][0]
    job = {'id':'fixture', 'output_directory':'run/jobs/fixture', 'frozen_files':[]}
    output = tmp_path/job['output_directory']
    output.mkdir(parents=True)
    artifact = output/'artifact.bin'
    artifact.write_bytes(b'preserved artifact')
    result = {'status':'completed','experiment_sha256':hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
              'artifacts':[{'path':artifact.relative_to(tmp_path).as_posix(),'sha256':digest(artifact)}]}
    (output/'result.json').write_text(json.dumps(result))
    queue_path = tmp_path/binding['queue_path']
    queue_path.write_text(json.dumps({'settings':{'output_root':'run'}, 'jobs':[job]}))
    binding['queue_sha256'] = digest(queue_path)
    before = (output/'result.json').read_bytes()
    controller, namespace, selected = load_controller(tmp_path, config, 'crossad')
    monkeypatch.setattr(sys, 'argv', ['controller','--queue',str(queue_path)])
    controller.main()
    state = json.loads((tmp_path/'run/status.json').read_text())
    assert state['state']=='completed' and state['jobs']=={'fixture':'completed'}
    assert (output/'result.json').read_bytes()==before
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError):
        controller.main()


def test_original_shared_slot_binding_reaches_compiled_controller_namespace(tmp_path, monkeypatch):
    config = fixture_root(tmp_path)
    controller, namespace, binding = load_controller(tmp_path, config, 'preflight')
    settings = {'minimum_available_memory_bytes':24,'minimum_commit_headroom_bytes':16}
    namespace['resource_snapshot'] = lambda: {'physical_available_bytes':24,'commit_available_bytes':16}
    namespace['bind_controller'](controller, tmp_path, settings)
    namespace['waited_lock'] = controller.waited_lock
    namespace['wait_for_headroom'] = controller.wait_for_headroom
    with namespace['waited_lock'](tmp_path/'experiments/runs/fm_heavy_recovery_cpu.lock'):
        assert namespace['wait_for_headroom'](settings)['commit_available_bytes']==16
    with pytest.raises(RuntimeError, match='owned shared slot'):
        namespace['wait_for_headroom'](settings)


def test_status_audit_recognizes_light_process_only_for_its_frozen_queue():
    from types import SimpleNamespace
    from scripts.flow_matching.summarize_tsad_execution import light_worker_matches
    process=SimpleNamespace(cmdline=lambda: [sys.executable,'-m','scripts.flow_matching.run_light_controller',
        '--runtime-config','configs/runtime/fm_light_controllers.v1.json','--controller','range_main'])
    assert light_worker_matches(process,'configs/experiments/fm_tab_range_execution.sparse.v2.json',ROOT)
    assert not light_worker_matches(process,'configs/experiments/fm_reflow_tab_range.sparse.v2.json',ROOT)
