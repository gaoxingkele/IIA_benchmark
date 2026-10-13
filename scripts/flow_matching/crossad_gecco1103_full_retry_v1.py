"""Full original CrossAD retry after a preserved, non-resumable commit abort.

Author/data citations and every original main/ablation obligation come from the
frozen 139-case queue. Only the selected output and admission reserve change.
"""
import copy
import hashlib
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller

CANONICAL = 'crossad_author__fresh_train__gecco__seed1103'
ORIGINAL_QUEUE = 'configs/experiments/fm_crossad_complete_queue.v7.json'
RUNTIME = 'configs/runtime/fm_light_controllers.v1.json'
NATIVE_QUEUE = 'configs/experiments/fm_crossad_gecco1103_full20_retry_queue.v1.json'
CONFIG = 'configs/runtime/fm_crossad_gecco1103_full20_retry.v1.json'
STATE_ROOT = 'experiments/runs/fm_crossad_gecco1103_full20_retry_v1'
EVIDENCE = 'projects/flow_matching_research/results/2026-10-13/crossad_gecco1103_full20_commit_retry_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Never overwrite a frozen retry artifact: ' + str(path))
        return
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def build_queue(original):
    queue = copy.deepcopy(original)
    ids = [j['id'] for j in queue['jobs']]
    if len(ids) != 139 or len(set(ids)) != 139 or ids.count(CANONICAL) != 1:
        raise ValueError('All 139 original main/release/ablation slots required')
    job = next(j for j in queue['jobs'] if j['id'] == CANONICAL)
    if (job['seed'] != 1103 or job['mode'] != 'fresh_train'
            or job['train_parameters']['train_epochs'] != 20
            or job['train_parameters']['batch_size'] != 128
            or job['train_parameters']['patience'] != 3):
        raise ValueError('Original full training recipe required')
    job['output_directory'] = STATE_ROOT + '/jobs/' + CANONICAL
    return queue


def verify_incident(root, incident, check_live=True):
    for pin in incident['source_receipts']:
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Preserved incident source changed: ' + pin['path'])
    for pin in incident['original_partial_artifacts']:
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Original partial evidence changed: ' + pin['path'])
    failure = read(root / incident['terminal_failure_path'])
    resources = failure['resources']
    if (failure['exit_code'] != 1 or resources['resource_guard_aborted'] is not True
            or resources['abort_resource_snapshot']['commit_available_bytes'] >= 8 * 1024 ** 3):
        raise ValueError('Actual original emergency commit abort required')
    trace = read(root / incident['training_trace_path'])
    if (trace['epochs_finished'] != 12 or trace['declared_max_epochs'] != 20
            or trace['training_forward_calls'] != 5184 or len(trace['validation_losses']) != 12):
        raise ValueError('Preserve actual completed 12-epoch history')
    output = root / incident['original_output_directory']
    if (output / 'result.json').exists() or (output / 'stages').exists():
        raise ValueError('Unexpected completed native pipeline; do not retrain')
    if (incident['optimizer_rng_resume_available'] is not False
            or incident['completed_budget_credit_for_fresh_retry'] != 0):
        raise ValueError('Weights alone cannot provide exact training continuation')
    archive = Path(incident['checkpoint_archive']['path'])
    if digest(archive) != incident['checkpoint_archive']['sha256']:
        raise ValueError('Preserved partial checkpoint archive changed')
    if check_live:
        import psutil
        old = incident['stopped_scientific_identity']
        for process in psutil.process_iter(['create_time', 'cmdline']):
            command = process.info['cmdline'] or []
            if (CANONICAL in command or (process.info['create_time'] == old['create_time']
                    and command == old['command'])):
                raise ValueError('Never duplicate a live original scientific process')
    return trace


def verify(config, root=ROOT, check_live=True):
    required = {ORIGINAL_QUEUE, RUNTIME, NATIVE_QUEUE,
        EVIDENCE + '/actual_nonresumable_commit_incident.json',
        'scripts/flow_matching/crossad_gecco1103_full_retry_v1.py',
        'scripts/flow_matching/run_crossad_gecco1103_full_retry_v1.py',
        'scripts/flow_matching/register_crossad_gecco1103_full_retry_v1.py'}
    if (config['original_queue'] != ORIGINAL_QUEUE or config['native_queue'] != NATIVE_QUEUE
            or {p['path'] for p in config['source_receipts']} != required):
        raise ValueError('Every original/dispatch/incident source binding required')
    for pin in config['source_receipts']:
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Frozen retry source changed: ' + pin['path'])
    original = read(root / config['original_queue'])
    queue = read(root / config['native_queue'])
    if queue != build_queue(original):
        raise ValueError('Only one output binding may change; retain every original field')
    if config['canonical_job_ids'] != [j['id'] for j in original['jobs']]:
        raise ValueError('Full original roster changed')
    if config['selected_original_ids'] != [CANONICAL]:
        raise ValueError('Other original slots remain owned by their existing queues')
    if (config['settings'] != dict(original['settings'], minimum_commit_headroom_bytes=40 * 1024 ** 3)
            or config['state_root'] != STATE_ROOT or config['runtime_config'] != RUNTIME
            or config['shared_lock'] != 'experiments/runs/fm_heavy_recovery_cpu.lock'):
        raise ValueError('Only stronger admission may change; emergency guard stays original')
    expected = config['execution_python']
    if expected != config['incident']['stopped_scientific_identity']['command'][0]:
        raise ValueError('Retain the actual pre-incident scientific interpreter')
    incident = config['incident']
    verify_incident(root, incident, check_live)
    sources = {p['path']: p['sha256'] for j in original['jobs'] for p in j['frozen_files']}
    for name, expected_sha in sources.items():
        if digest(root / name) != expected_sha:
            raise ValueError('Original scientific source/data changed: ' + name)
    runtime = read(root / config['runtime_config'])
    controller, api, _ = load_controller(root, runtime, 'crossad')
    job = next(j for j in queue['jobs'] if j['id'] == CANONICAL)
    proof = controller.native_proof(root, job, original['settings'])
    if proof is None or proof != config['native_preflight_binding']:
        raise ValueError('Actual original full-batch preflight required')
    return queue, job, api


def verify_completed(root, job, api):
    output = root / job['output_directory']
    receipt = read(output / 'result.json')
    expected = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if (receipt['status'] != 'completed' or receipt['id'] != job['id']
            or receipt['experiment_sha256'] != expected or receipt['seed'] != job['seed']
            or receipt['mode'] != 'fresh_train' or receipt['dataset'] != job['dataset']):
        raise ValueError('Actual canonical original result binding required')
    if (receipt['details']['declared_train_parameters'] != job['train_parameters']
            or not 1 <= receipt['details']['epochs_completed'] <= job['train_parameters']['train_epochs']):
        raise ValueError('Original full maximum budget and author early stopping required')
    for stage in job['stages']:
        value = read(output / 'stages' / stage['id'] / 'receipt.json')
        if value['status'] != 'completed':
            raise ValueError('Every original native stage must complete')
        api['verify_artifacts'](value)
    api['verify_artifacts'](receipt)
    if receipt['strict_TAB_result'] is not False or receipt['paper_equivalence'] is not False:
        raise ValueError('Native flat-data completion is not TAB/paper equivalence')
    return receipt
