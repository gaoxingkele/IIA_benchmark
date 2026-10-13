"""Same canonical PSM recipe after the preserved pre-first-epoch commit abort.

The full139 CrossAD queue supplies the literature, data and method bindings.
Only this output path and admission reserve change; no training credit is taken
from the interrupted attempt. The existing author runner and guards are reused.
"""
from types import FunctionType

from scripts.flow_matching import crossad_gecco1103_full_retry_v1 as previous
from scripts.flow_matching import run_crossad_gecco1103_full_retry_v1 as dispatch

ROOT = previous.ROOT
CANONICAL = 'crossad_author__fresh_train__psm__seed1103'
STATE = 'experiments/runs/fm_crossad_psm1103_full20_retry_v1'


def build_queue(original):
    function = FunctionType(previous.build_queue.__code__,
        dict(previous.build_queue.__globals__, CANONICAL=CANONICAL, STATE_ROOT=STATE))
    return function(original)


def verify(config, root=ROOT, check_live=True):
    required = {previous.ORIGINAL_QUEUE, previous.RUNTIME,
        'configs/experiments/fm_crossad_psm1103_full20_retry_queue.v1.json',
        'scripts/flow_matching/crossad_psm1103_full_retry_v1.py',
        'scripts/flow_matching/crossad_gecco1103_full_retry_v1.py',
        'scripts/flow_matching/run_crossad_gecco1103_full_retry_v1.py',
        'tests/test_crossad_psm_full_retry_v1.py',
        'projects/flow_matching_research/results/2026-10-13/crossad_psm1103_full20_retry_v1/incident.json'}
    if (config['original_queue'] != previous.ORIGINAL_QUEUE or config['runtime_config'] != previous.RUNTIME
            or {p['path'] for p in config['source_receipts']} != required):
        raise ValueError('Every exact original/incident/dispatch binding is required')
    for pin in config['source_receipts']:
        if previous.digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Frozen PSM retry binding changed: ' + pin['path'])
    original = previous.read(root / config['original_queue'])
    queue = previous.read(root / config['native_queue'])
    if (queue != build_queue(original) or config['canonical_job_ids'] != [j['id'] for j in original['jobs']]
            or config['selected_original_ids'] != [CANONICAL]
            or config['settings'] != dict(original['settings'], minimum_commit_headroom_bytes=40 * 1024 ** 3)
            or config['state_root'] != STATE
            or config['shared_lock'] != 'experiments/runs/fm_heavy_recovery_cpu.lock'):
        raise ValueError('All139 slots and exact original science/guards required')
    incident = config['incident']
    failure = previous.read(root / incident['terminal_failure_path'])
    old = next(j for j in original['jobs'] if j['id'] == CANONICAL)
    output = root / old['output_directory']
    for pin in incident['original_partial_artifacts']:
        if previous.digest(root / pin['path']) != pin['sha256']:
            raise ValueError('Preserved PSM partial artifact changed')
    if (failure['exit_code'] != 1 or failure['resources']['resource_guard_aborted'] is not True
            or failure['resources']['abort_resource_snapshot']['commit_available_bytes'] >= 8 * 1024 ** 3
            or (output / 'result.json').exists() or any(output.rglob('*.pth'))
            or incident['credited_previous_epochs'] != 0 or incident['resume_available'] is not False):
        raise ValueError('Actual terminal abort without a resumable checkpoint required')
    if check_live:
        import psutil
        for process in psutil.process_iter(['name', 'cmdline']):
            command = process.info['cmdline'] or []
            if ('python' in (process.info['name'] or '').lower()
                    and CANONICAL in command and '--job-id' in command):
                raise ValueError('Do not duplicate a live canonical PSM worker')
    if config['execution_python'] != incident['stopped_identity']['command'][0]:
        raise ValueError('Actual pre-incident numerical interpreter must remain')
    sources = {p['path']: p['sha256'] for j in original['jobs'] for p in j['frozen_files']}
    for path, digest in sources.items():
        if previous.digest(root / path) != digest:
            raise ValueError('Original scientific source/data binding changed: ' + path)
    runtime = previous.read(root / config['runtime_config'])
    controller, api, _ = previous.load_controller(root, runtime, 'crossad')
    job = next(j for j in queue['jobs'] if j['id'] == CANONICAL)
    proof = controller.native_proof(root, job, original['settings'])
    if proof is None or proof != config['native_preflight_binding']:
        raise ValueError('Original complete native batch preflight required')
    return queue, job, api


def main():
    namespace = dict(dispatch.main.__globals__, CANONICAL=CANONICAL, verify=verify)
    FunctionType(dispatch.main.__code__, namespace)()


if __name__ == '__main__':
    main()
