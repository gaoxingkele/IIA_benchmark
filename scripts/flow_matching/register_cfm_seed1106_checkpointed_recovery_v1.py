"""Preserve the unsaved223-epoch incident; register full400 with complete state."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.run_cfm_checkpointed_seed1106_parent_v1 import ROOT, parent_namespace

QUEUE = 'configs/experiments/fm_cfm_node1106_complete_epoch_full_retry_queue.v10.json'
CONFIG = 'configs/runtime/fm_cfm_node1106_complete_epoch_full_retry.v14.json'
REPLAY = 'configs/runtime/fm_cfm_completion_replay_queue.v3.json'
RESULT = 'projects/flow_matching_research/results/2026-10-13/cfm_seed1106_full400_checkpointed_recovery_v1'


def exclusive(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def register():
    import psutil
    values = parent_namespace()
    read, sha, verify = [values['original'].__dict__[k] for k in ('read', 'sha', 'verify')]
    def pin(path): return dict(path=path, sha256=sha(ROOT / path))
    identity = 'cfm_ts_original__main_text__node_e400__pendulum__seed1106'
    source_path = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
    failed_path = 'configs/experiments/fm_cfm_node1106_shared_commit_full_retry_queue.v8.json'
    source, failed_queue = read(ROOT / source_path), read(ROOT / failed_path)
    verify(source);verify(failed_queue)
    canonical = next(j for j in source['jobs'] if j['id'] == identity)
    failed = next(j for j in failed_queue['jobs'] if j['id'] == identity)
    receipt_path = failed_queue['state_root'] + '/receipts/' + identity + '.json'
    progress_path = failed['output_directory'] + '/progress.json'
    receipt, progress = read(ROOT / receipt_path), read(ROOT / progress_path)
    if (receipt['exit_code'] == 0 or receipt['complete']
            or not receipt['resource_observations']['resource_guard_aborted']
            or progress['epoch'] != 223 or progress['optimizer_updates'] != 223
            or progress['required_epochs'] != 400
            or values['canonical_science'](failed) != values['canonical_science'](canonical)):
        raise ValueError('Require actual resource-aborted223-of400 canonical seed1106')
    if psutil.pid_exists(27288) and psutil.Process(27288).create_time() == 1791780560.1530883:
        raise ValueError('Never register over live scientific training')
    for directory in (ROOT / failed['output_directory'], ROOT / failed['artifact_directory']):
        if directory.exists() and any(p.is_file() and p.suffix in ('.pt', '.pth', '.pkl') for p in directory.rglob('*')):
            raise ValueError('Actual native checkpoint requires restoration audit, not fresh restart')
    diagnostic_path = 'projects/flow_matching_research/results/2026-10-12/cfm_complete_epoch_full400_retry_v1/native_validation_v2/actual_guarded_native_checkpoint_diagnostic_receipt.json'
    diagnostic = read(ROOT / diagnostic_path)
    if diagnostic['exit_code'] != 0 or not diagnostic['diagnostic_checks_passed'] or diagnostic['resource_guard_aborted']:
        raise ValueError('Successful unchanged full-architecture checkpoint diagnostic required')
    previous_path = 'configs/runtime/fm_cfm_completion_replay_queue.v2.json'
    from scripts.flow_matching.run_cfm_completion_replay_queue_v2 import original as replay_source, allowed_idle
    previous = read(ROOT / previous_path);replay_source.verify(previous)
    owner_state_path = previous['state_root'] + '/status.json'
    owner_state = read(ROOT / owner_state_path)
    actor = psutil.Process(owner_state['pid'])
    children = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in actor.children(recursive=True)]
    if (actor.create_time() != owner_state['create_time'] or actor.cmdline()[-1] != previous_path
            or not allowed_idle(owner_state, children)):
        raise ValueError('Require the verified idle all285 replay owner; never stop native work')
    for path in (QUEUE, CONFIG, REPLAY, RESULT):
        if (ROOT / path).exists(): raise FileExistsError('Preserve prior registrations: ' + path)
    queue = deepcopy(source)
    job = next(j for j in queue['jobs'] if j['id'] == identity)
    job.update(id=identity + '__checkpointed_full_retry_v10', recovery_original_job_id=identity,
        output_directory='experiments/runs/fm_cfm_node1106_complete_epoch_full_retry_v10/jobs/' + identity,
        artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_node1106_complete_epoch_full_retry_v10/' + identity)
    policy = deepcopy(failed_queue['resource_policy'])
    policy['minimum_commit_headroom_bytes'] = 34359738368
    queue.update(state_root='experiments/runs/fm_cfm_node1106_complete_epoch_full_retry_v10',
        resource_policy=policy, recovery_original_queue=source_path,
        recovery_original_queue_sha256=sha(ROOT / source_path), recovery_job_ids=[job['id']],
        recovery_selection='Same canonical seed1106; unsaved223 epochs get zero credit. Full400 and all500 test points retained; admission raised to32GiB, emergency16GiB retained.')
    sources = [source_path, failed_path, receipt_path, progress_path, diagnostic_path,
        'scripts/flow_matching/run_cfm_checkpointed_seed1106_job_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_seed1106_parent_v1.py',
        'scripts/flow_matching/register_cfm_seed1106_checkpointed_recovery_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_full_retry_v1.py',
        'scripts/flow_matching/run_cfm_checkpointed_light_parent_v2.py',
        'scripts/flow_matching/cfm_complete_epoch_checkpoint_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'tests/test_fm_cfm_seed1106_checkpointed_recovery_v1.py']
    queue['source_receipts'] += [pin(p) for p in sources]
    assert len(queue['jobs']) == 285
    assert all(a == b for a, b in zip(source['jobs'], queue['jobs']) if a['id'] != identity)
    exclusive(QUEUE, queue)
    config = dict(queue=QUEUE, job_id=job['id'], canonical_job=canonical,
        state_root=queue['state_root'], family_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',
        resource_policy=policy, checkpoint_interval=10, resume_full_epoch_checkpoint=None,
        resume_full_epoch_checkpoint_sha256=None,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        source_receipts=[pin(p) for p in sources + [QUEUE]], citation=read(ROOT / source['protocol'])['citation'],
        reproduction_status='executable_paper_derived_reconstruction_not_author_equivalent',
        original_failed_attempt=failed, preserved_failed_terminal=receipt_path,
        discarded_unsaved223_epochs_not_counted=True)
    values['verify_registration'](config);exclusive(CONFIG, config)
    replay = deepcopy(previous)
    route = next(r for r in replay['routes'] if r['original_canonical_id'] == identity)
    route.update(job=job, result_path=job['output_directory'] + '/result.json',
        checkpointed_training_terminal=config['state_root'] + '/receipt.json')
    replay.update(runtime_configuration=REPLAY, state_root='experiments/runs/fm_cfm_completion_replay_queue_v3',
        child_state_root='experiments/runs/fm_cfm_completion_replay_queue_v3/children',
        child_config_root='configs/reproducibility/fm_cfm_completion_replay_queue_v3',
        artifact_root='F:/aicoding/IIA_Data/experiments/cfm_completion_replay_queue_v3',
        previous_owner=dict(pid=actor.pid, create_time=actor.create_time(), command=actor.cmdline()),
        previous_state_path=owner_state_path, poll_seconds=10)
    replay['source_receipts'] = [pin(p) for p in sorted({r['path'] for r in previous['source_receipts']} | set(sources + [QUEUE, CONFIG, previous_path]))]
    replay_source.verify(replay);exclusive(REPLAY, replay)
    exclusive(RESULT + '/actual_resource_abort_and_full400_registration.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), actual_terminal=receipt,
        failed_worker=dict(pid=27288, create_time=1791780560.1530883),
        retained_unsaved_epochs=223, retained_partial_epochs_credited=0,
        original_required_epochs=400, actual_fresh_training_completed=False,
        checkpoint_interval=10, complete_epoch_checkpoints_required=40,
        minimum_commit_headroom_bytes=34359738368, emergency_commit_headroom_bytes=17179869184,
        source_solver_and_optimizer_unchanged=True, source_data_seed_and_full500_test_points_unchanged=True,
        canonical_job=canonical, replay_route=route, original_CFM_jobs=285,
        preserved_verified_original_models=85, other284_replay_routes_unchanged=all(a==b for a,b in zip(previous['routes'],replay['routes']) if a['original_canonical_id']!=identity),
        configuration_receipts=[pin(p) for p in (QUEUE, CONFIG, REPLAY)], previous_idle_replay_owner=replay['previous_owner'],
        original_scope=config['original_scope'], benchmark_performance=False,
        author_equivalence_certified=False, all_original_experiments_complete=False))
    print(dict(queue=QUEUE,configuration=CONFIG,replay=REPLAY,original_epochs=400,retained_partial_epochs_credited=0))


if __name__ == '__main__':
    register()
