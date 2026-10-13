import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.flow_matching import crossad_gecco1103_full_retry_v1 as retry


def original_queue():
    job = {'id': retry.CANONICAL, 'seed': 1103, 'mode': 'fresh_train', 'dataset': 'GECCO',
        'train_parameters': {'train_epochs': 20, 'batch_size': 128, 'patience': 3},
        'model_parameters': {'seq_len': 128, 'dropout': .1}, 'raw_sha256': 'raw',
        'frozen_files': [], 'runner': 'original_runner.py', 'output_directory': 'original',
        'stages': [{'id': s} for s in ('weights', 'author_test', 'author_spot', 'native_control')]}
    jobs = [job] + [dict(copy.deepcopy(job), id='original-slot-' + str(i)) for i in range(138)]
    return {'jobs': jobs, 'settings': {'minimum_available_memory_bytes': 24 * 1024 ** 3,
        'minimum_commit_headroom_bytes': 16 * 1024 ** 3,
        'emergency_commit_headroom_bytes': 8 * 1024 ** 3, 'cpu_threads': 2}}


def test_full_roster_keeps_every_other_method_and_ablation_binding():
    old = original_queue()
    new = retry.build_queue(old)
    assert new['settings'] == old['settings']
    assert new['jobs'][1:] == old['jobs'][1:]
    assert len(new['jobs']) == 139
    before, after = old['jobs'][0], new['jobs'][0]
    assert after['id'] == before['id'] and after['seed'] == 1103
    assert {k: v for k, v in after.items() if k != 'output_directory'} == {
        k: v for k, v in before.items() if k != 'output_directory'}
    assert after['output_directory'] != before['output_directory']


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'seed', 'epochs', 'batch', 'patience'])
def test_reject_shortened_scope_or_changed_original_budget(kind):
    old = original_queue()
    if kind == 'missing':
        old['jobs'].pop()
    elif kind == 'duplicate':
        old['jobs'][-1]['id'] = old['jobs'][1]['id']
    elif kind == 'seed':
        old['jobs'][0]['seed'] = 1104
    else:
        key = {'epochs': 'train_epochs', 'batch': 'batch_size', 'patience': 'patience'}[kind]
        old['jobs'][0]['train_parameters'][key] -= 1
    with pytest.raises(ValueError):
        retry.build_queue(old)


def config_fixture(root, monkeypatch):
    old = original_queue()
    retry.freeze(root / retry.ORIGINAL_QUEUE, old)
    retry.freeze(root / retry.NATIVE_QUEUE, retry.build_queue(old))
    retry.freeze(root / retry.RUNTIME, {})
    folder = root / retry.EVIDENCE
    failure = folder / 'original_failure.json'
    trace = folder / 'original_training_trace.json'
    retry.freeze(failure, {'exit_code': 1, 'resources': {'resource_guard_aborted': True,
        'abort_resource_snapshot': {'commit_available_bytes': 7297990656}}})
    retry.freeze(trace, {'epochs_finished': 12, 'declared_max_epochs': 20,
        'training_forward_calls': 5184, 'validation_losses': [.6] * 12})
    archive = root / 'partial_best_weights.pth'
    archive.write_bytes(b'archived-real-checkpoint-binding-fixture')
    command = ['original-python.exe', '-u', 'original_runner.py', '--job-id', retry.CANONICAL]
    incident = {'source_receipts': [{'path': str(p.relative_to(root)), 'sha256': retry.digest(p)}
        for p in (failure, trace)], 'original_partial_artifacts': [],
        'terminal_failure_path': str(failure.relative_to(root)),
        'training_trace_path': str(trace.relative_to(root)), 'original_output_directory': 'original',
        'optimizer_rng_resume_available': False, 'completed_budget_credit_for_fresh_retry': 0,
        'checkpoint_archive': {'path': str(archive), 'sha256': retry.digest(archive)},
        'stopped_scientific_identity': {'create_time': 1, 'command': command}}
    incident_path = folder / 'actual_nonresumable_commit_incident.json'
    retry.freeze(incident_path, incident)
    script_paths = ['scripts/flow_matching/' + name + '.py' for name in (
        'crossad_gecco1103_full_retry_v1', 'run_crossad_gecco1103_full_retry_v1',
        'register_crossad_gecco1103_full_retry_v1')]
    for path in script_paths:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'frozen-original-controller-fixture')
    proof = {'native_batch_shape': [128, 128, 9]}
    monkeypatch.setattr(retry, 'load_controller', lambda *args: (
        SimpleNamespace(native_proof=lambda *args: proof), {'verify_artifacts': lambda x: None}, None))
    paths = [retry.ORIGINAL_QUEUE, retry.NATIVE_QUEUE, retry.RUNTIME,
        str(incident_path.relative_to(root)).replace('\\', '/')] + script_paths
    config = {'original_queue': retry.ORIGINAL_QUEUE, 'native_queue': retry.NATIVE_QUEUE,
        'canonical_job_ids': [j['id'] for j in old['jobs']], 'selected_original_ids': [retry.CANONICAL],
        'settings': dict(old['settings'], minimum_commit_headroom_bytes=40 * 1024 ** 3),
        'execution_python': command[0], 'runtime_config': retry.RUNTIME, 'state_root': retry.STATE_ROOT,
        'shared_lock': 'experiments/runs/fm_heavy_recovery_cpu.lock', 'incident': incident,
        'native_preflight_binding': proof,
        'source_receipts': [{'path': p, 'sha256': retry.digest(root / p)} for p in paths]}
    return config


def test_all139_verified_and_only_admission_is_stronger(tmp_path, monkeypatch):
    config = config_fixture(tmp_path, monkeypatch)
    queue, job, _ = retry.verify(config, tmp_path, check_live=False)
    assert len(queue['jobs']) == 139 and job['id'] == retry.CANONICAL
    assert queue['settings']['minimum_commit_headroom_bytes'] == 16 * 1024 ** 3
    assert config['settings']['minimum_commit_headroom_bytes'] == 40 * 1024 ** 3
    assert config['settings']['emergency_commit_headroom_bytes'] == 8 * 1024 ** 3


@pytest.mark.parametrize('kind', ['guard', 'cpu', 'lock', 'selected', 'roster', 'credit', 'source', 'archive'])
def test_reject_weakened_guard_duplicate_slot_or_false_resume_credit(tmp_path, monkeypatch, kind):
    config = config_fixture(tmp_path, monkeypatch)
    if kind == 'guard':
        config['settings']['emergency_commit_headroom_bytes'] -= 1
    elif kind == 'cpu':
        config['settings']['cpu_threads'] = 1
    elif kind == 'lock':
        config['shared_lock'] = 'different.lock'
    elif kind == 'selected':
        config['selected_original_ids'].append('another-original-slot')
    elif kind == 'roster':
        config['canonical_job_ids'].pop()
    elif kind == 'credit':
        config['incident']['completed_budget_credit_for_fresh_retry'] = 12
    elif kind == 'source':
        config['source_receipts'].pop()
    else:
        Path(config['incident']['checkpoint_archive']['path']).write_bytes(b'wrong-checkpoint')
    with pytest.raises(ValueError):
        retry.verify(config, tmp_path, check_live=False)


def test_live_original_identity_prevents_duplicate_training(tmp_path, monkeypatch):
    import psutil
    config = config_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(psutil, 'process_iter', lambda attrs: [SimpleNamespace(info={
        'create_time': 1, 'cmdline': config['incident']['stopped_scientific_identity']['command']})])
    with pytest.raises(ValueError, match='live original'):
        retry.verify(config, tmp_path)


def test_partial_weights_or_one_successful_stage_cannot_claim_native_completion(tmp_path):
    job = retry.build_queue(original_queue())['jobs'][0]
    output = tmp_path / job['output_directory']
    value = {'status': 'completed', 'id': job['id'], 'seed': 1103, 'mode': 'fresh_train', 'dataset': 'GECCO',
        'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        'details': {'declared_train_parameters': job['train_parameters'], 'epochs_completed': 20},
        'strict_TAB_result': False, 'paper_equivalence': False}
    retry.freeze(output / 'result.json', value)
    retry.freeze(output / 'stages/weights/receipt.json', {'status': 'completed'})
    with pytest.raises(FileNotFoundError):
        retry.verify_completed(tmp_path, job, {'verify_artifacts': lambda x: None})


def test_frozen_output_and_raw_incident_bytes_never_overwritten(tmp_path):
    path = tmp_path / 'receipt.json'
    retry.freeze(path, {'original': True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Never overwrite'):
        retry.freeze(path, {'original': False})
    assert path.read_bytes() == before
