"""Archive the actual CrossAD12/20 commit incident and freeze a full139 retry."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import psutil
from scripts.flow_matching.crossad_gecco1103_full_retry_v1 import (
    ROOT, CANONICAL, ORIGINAL_QUEUE, RUNTIME, NATIVE_QUEUE, CONFIG, STATE_ROOT,
    EVIDENCE, read, freeze, build_queue, verify,
)
from scripts.flow_matching.run_light_controller import digest, load_controller

OLD_IDENTITY = {'pid': 36568, 'create_time': 1791780874.360876,
    'command': [r'C:\Users\10175\AppData\Local\Programs\Python\Python312\python.exe',
        '-u', str(ROOT / 'scripts/flow_matching/run_crossad_memory_v2_job.py'),
        '--queue', str(ROOT / ORIGINAL_QUEUE), '--job-id', CANONICAL]}
ARCHIVE = Path('F:/aicoding/IIA_Data/experiments/verified_incident_checkpoint_archives_v1') / CANONICAL / 'epoch012_best_weights_only.pth'


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != digest(source):
            raise ValueError('Never overwrite preserved evidence: ' + str(destination))
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    if digest(destination) != digest(source):
        raise ValueError('Evidence copy failed checksum verification')


def main():
    if (ROOT / CONFIG).exists():
        verify(read(ROOT / CONFIG))
        print(json.dumps({'already_registered': True, 'configuration': CONFIG}))
        return
    for process in psutil.process_iter(['create_time', 'cmdline']):
        command = process.info['cmdline'] or []
        if CANONICAL in command or (process.info['create_time'] == OLD_IDENTITY['create_time']
                and command == OLD_IDENTITY['command']):
            raise ValueError('Original scientific process remains live')
    original = read(ROOT / ORIGINAL_QUEUE)
    original_job = next(j for j in original['jobs'] if j['id'] == CANONICAL)
    output = ROOT / original_job['output_directory']
    failure = read(output / 'failure.json')
    trace = read(output / 'training_trace.json')
    assert failure['exit_code'] == 1 and failure['resources']['resource_guard_aborted']
    assert failure['resources']['abort_resource_snapshot']['commit_available_bytes'] == 7297990656
    assert trace['epochs_finished'] == 12 and trace['training_forward_calls'] == 5184
    assert trace['declared_max_epochs'] == 20 and not (output / 'result.json').exists()
    assert not (output / 'stages').exists()
    source = ROOT / original['settings']['runtime_root'] / 'exp/exp_anomaly_detection.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    stopping = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EarlyStopping')
    save = next(n for n in stopping.body if isinstance(n, ast.FunctionDef) and n.name == 'save_checkpoint')
    calls = [n for n in ast.walk(save) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' and n.func.attr == 'save']
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == 'model.state_dict()'
    checkpoint = output / 'workspace/configs/GECCO/checkpoints_0/checkpoint.pth'
    copy_exact(checkpoint, ARCHIVE)
    evidence = ROOT / EVIDENCE
    snapshot_names = ('failure.json', 'resource_usage.json', 'training_trace.json',
        'native_preflight_binding.json', 'stdout.log', 'stderr.log')
    for name in snapshot_names:
        copy_exact(output / name, evidence / ('original_' + name))
    copy_exact(output / 'workspace/configs/GECCO/checkpoints_0/basic_info.csv', evidence / 'original_basic_info.csv')
    copy_exact(ROOT / 'experiments/runs/fm_crossad_author_v2/status.json', evidence / 'original_controller_status_snapshot.json')
    attributes = evidence / '.gitattributes'
    attributes.parent.mkdir(parents=True, exist_ok=True)
    with attributes.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('*.log -diff -text\n')
    immutable = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)}
        for p in sorted(evidence.iterdir()) if p.is_file()]
    original_artifacts = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size}
        for p in sorted(output.rglob('*')) if p.is_file()]
    incident = {'captured_utc': datetime.now(timezone.utc).isoformat(),
        'stopped_scientific_identity': OLD_IDENTITY, 'verified_no_matching_live_scientific_process': True,
        'original_output_directory': original_job['output_directory'],
        'terminal_failure_path': EVIDENCE + '/original_failure.json',
        'training_trace_path': EVIDENCE + '/original_training_trace.json',
        'source_receipts': immutable,
        'original_partial_artifacts': original_artifacts,
        'checkpoint_archive': {'path': str(ARCHIVE), 'sha256': digest(ARCHIVE), 'bytes': ARCHIVE.stat().st_size},
        'checkpoint_save_source': {'path': source.relative_to(ROOT).as_posix(), 'sha256': digest(source),
            'argument_ast': ast.unparse(calls[0].args[0])},
        'optimizer_rng_resume_available': False,
        'preserved_historical_completed_epochs': 12,
        'partial_epoch13_last_logged_iteration': 200,
        'exact_unsaved_epoch13_optimizer_updates': None,
        'completed_budget_credit_for_fresh_retry': 0,
        'recovery_policy': 'Same canonical seed and original full20 maximum/patience3, fresh full pipeline in separate output; stronger40GiB admission, unchanged8GiB emergency guard. No exact optimizer/RNG resume claim.',
        'additional_independent_seed_slots': 0, 'all_original_experiments_complete': False}
    freeze(evidence / 'actual_nonresumable_commit_incident.json', incident)
    queue = build_queue(original)
    freeze(ROOT / NATIVE_QUEUE, queue)
    controller, _, _ = load_controller(ROOT, read(ROOT / RUNTIME), 'crossad')
    proof = controller.native_proof(ROOT, original_job, original['settings'])
    assert proof is not None
    pins = {p: digest(ROOT / p) for p in [ORIGINAL_QUEUE, RUNTIME, NATIVE_QUEUE,
        EVIDENCE + '/actual_nonresumable_commit_incident.json',
        'scripts/flow_matching/crossad_gecco1103_full_retry_v1.py',
        'scripts/flow_matching/run_crossad_gecco1103_full_retry_v1.py',
        'scripts/flow_matching/register_crossad_gecco1103_full_retry_v1.py']}
    config = {'original_queue': ORIGINAL_QUEUE, 'native_queue': NATIVE_QUEUE,
        'canonical_job_ids': [j['id'] for j in original['jobs']], 'selected_original_ids': [CANONICAL],
        'settings': dict(original['settings'], minimum_commit_headroom_bytes=40 * 1024 ** 3),
        'execution_python': OLD_IDENTITY['command'][0], 'runtime_config': RUNTIME,
        'state_root': STATE_ROOT, 'shared_lock': 'experiments/runs/fm_heavy_recovery_cpu.lock',
        'native_preflight_binding': proof, 'incident': incident,
        'source_receipts': [{'path': p, 'sha256': h} for p, h in sorted(pins.items())],
        'boundary': 'Retain full139 canonical obligations; select only actual failed GECCO1103. Other138 original output bindings unchanged. Same scientific interpreter, original author memory-v2 runner, full batches/loss/Adam/LR/patience/native evaluators/source pins. No added seed, no completed-budget credit from non-resumable partial weights, no automatic repeated failed attempt, no TAB/paper equivalence.'}
    verify(config)
    freeze(ROOT / CONFIG, config)
    print(json.dumps({'configuration': CONFIG, 'required_original_jobs': 139,
        'selected_original_ids': [CANONICAL], 'partial_checkpoint_archive_sha256': digest(ARCHIVE)}))


if __name__ == '__main__':
    main()
