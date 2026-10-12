"""Register full original Sine1103 training and retain the complete45-case replay."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


RECOVERY = 'configs/reproducibility/fm_timegan_sine_checkpointed_stage_recovery.2026-10-12_v4.json'
REPLAY = 'configs/runtime/fm_timegan_full45_completion_replay.v4.json'
RESULT = 'projects/flow_matching_research/results/2026-10-12/timegan_sine_checkpointed_full50000_recovery_v4'


def exclusive(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pin(path):
    return dict(path=str(path), sha256=sha(ROOT / path))


def register():
    import psutil
    from scripts.flow_matching.timegan_joint_checkpoint_observer_v1 import checkpoint_resume_compiler
    from scripts.flow_matching.run_timegan_full45_completion_replay_v3 import runtime_namespace, handoff_phase_allowed
    previous_recovery = 'configs/reproducibility/fm_timegan_sine_stage_recovery.2026-10-12_v3.json'
    previous_replay = 'configs/runtime/fm_timegan_full45_completion_replay.v3.json'
    old, replay_old = read(ROOT / previous_recovery), read(ROOT / previous_replay)
    queue = read(ROOT / old['original_queue'])
    terminal_path = old['state_root'] + '/receipt.json'
    previous_state_path = old['state_root'] + '/status.json'
    terminal, old_state = read(ROOT / terminal_path), read(ROOT / previous_state_path)
    progress_path = old['output_directory'] + '/progress.json'
    progress = read(ROOT / progress_path)
    if (terminal['exit_code'] == 0 or terminal['full_native_recovery_result_verified']
            or not terminal['resource_guard_aborted'] or old_state['active_pid'] is not None
            or progress['full_original_training_complete'] or progress['full_evaluation_complete']
            or progress['optimizer_updates']['E0'] != 50000 or progress['optimizer_updates']['GS'] != 50000):
        raise ValueError('Require the authoritative incomplete native Sine stage recovery')
    old_worker = dict(pid=34292, create_time=1791770428.680258)
    if psutil.pid_exists(old_worker['pid']) and psutil.Process(old_worker['pid']).create_time() == old_worker['create_time']:
        raise ValueError('Never replace an actually live scientific training worker')
    old_output = ROOT / old['output_directory']
    if any(old_output.rglob('*.index')) or any(old_output.rglob('*.meta')):
        raise ValueError('An existing joint checkpoint needs state auditing instead of discarding progress')
    for path in (RECOVERY, REPLAY, RESULT):
        if (ROOT / path).exists():
            raise FileExistsError('Preserve every earlier recovery/registration: ' + path)
    sources = {item['path'] for item in old['recovery_source_receipts']}
    sources.update([previous_recovery, previous_replay, terminal_path, previous_state_path, progress_path])
    sources.update(['scripts/flow_matching/' + name for name in (
        'timegan_joint_checkpoint_observer_v1.py', 'run_timegan_checkpointed_stage_recovery_v4.py',
        'run_timegan_full45_completion_replay_v3.py', 'register_timegan_checkpointed_sine_recovery_v4.py',
        'cached_resource_snapshot_v1.py')])
    sources.update(['tests/test_fm_timegan_joint_checkpoint_observer_v1.py',
                    'tests/test_fm_timegan_full45_checkpointed_route_v3.py'])
    recovery = deepcopy(old)
    recovery.update(output_directory='F:/aicoding/IIA_Data/experiments/timegan_sine_checkpointed_stage_recovery_v4/jobs/timegan_native__gru__sine__seed1103',
        state_root='experiments/runs/fm_timegan_sine_checkpointed_stage_recovery_v4',
        original_scope=replay_old['original_scope'], original_training_settings=queue['settings'],
        joint_checkpoint_interval=1000, joint_iterations=50000, checkpoint_full_variable_count=284,
        periodic_checkpoints_do_not_replace_original23_completion_checkpoints=True,
        failed_previous_attempt_preserved=dict(configuration=previous_recovery, terminal=terminal_path,
            output_directory=old['output_directory'], last_reported_optimizer_updates=progress['optimizer_updates'],
            native_joint_checkpoint_present=False, old_worker_identity_verified_not_live=old_worker),
        recovery_source_receipts=[pin(p) for p in sorted(sources)],
        boundary='Both actual50000-update pretraining states remain independently verified and preserved. Unsaved joint updates from both failed attempts receive no training credit. Execute all50000 original joint iterations again from exact supervised state, retaining source128batch/24hidden/3layers/full10000x24x5 data and all20 original full evaluators. Add observation-only full284-variable model/optimizer and NumPy/Python RNG checkpoints at50 complete1000-iteration boundaries; saved values/RNG/counters must be bitwise unchanged by observation. The original23 completion checkpoints,21 graph seeds, source budgets/guards, all45 TimeGAN cases and whole45paper681method273task scope are unchanged.')
    checkpoint_resume_compiler()(recovery)
    exclusive(RECOVERY, recovery)
    state_path = replay_old['state_root'] + '/status.json'
    state = read(ROOT / state_path)
    actor = psutil.Process(state['pid'])
    command = actor.cmdline()
    descendants = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in actor.children(recursive=True)]
    if (actor.create_time() != state['create_time'] or command[-1] != previous_replay
            or 'scripts.flow_matching.run_timegan_full45_completion_replay_v2' not in command
            or not handoff_phase_allowed('original', state, False, descendants)):
        raise ValueError('Require the exact idle all45 replay owner, without native children/history')
    replay = deepcopy(replay_old)
    replay.update(runtime_configuration=REPLAY,
        state_root='experiments/runs/fm_timegan_full45_completion_replay_v4',
        output_root='projects/flow_matching_research/results/2026-10-12/timegan_full45_completion_replay_v4/completed_models',
        previous_replay_owner=dict(pid=actor.pid, create_time=actor.create_time(), command=command),
        previous_replay_state_path=state_path)
    route = next(r for r in replay['routes'] if r['canonical_job']['id'] == old['canonical_job_id'])
    route.update(execution_job=dict(old['canonical_job'], output_directory=recovery['output_directory']),
        recovery_configuration=RECOVERY, result_path=recovery['output_directory']+'/result.json',
        native_completion_receipts=[dict(path=recovery['state_root']+'/receipt.json',
                                        successful_field='full_native_recovery_result_verified')])
    sources.update(p['path'] for p in replay_old['source_receipts'])
    sources.add(RECOVERY)
    replay['source_receipts'] = [pin(p) for p in sorted(sources)]
    replay['boundary'] += ' Sine1103 now routes to full50000 checkpointed joint recovery; all44 other routes, all23 checkpoint bitwise restores and all20 evaluators remain unchanged. Only the verified idle controller is handed off to reuse cached host telemetry; no native child is stopped.'
    runtime_namespace()['verify'](replay)
    exclusive(REPLAY, replay)
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(), recovery_configuration=RECOVERY,
        recovery_configuration_sha256=sha(ROOT/RECOVERY), replay_configuration=REPLAY,
        replay_configuration_sha256=sha(ROOT/REPLAY), original_scope=replay_old['original_scope'],
        original_TimeGAN_cases=45, other44_replay_routes_unchanged=all(
            a == b for a,b in zip(replay_old['routes'], replay['routes']) if a['canonical_job']['id'] != old['canonical_job_id']),
        full_joint_iterations=50000, full_evaluator_trainings=20, original_completion_checkpoints=23,
        complete_native_joint_checkpoint_count=50, every_native_variable_saved=284,
        source_checkpointed_training_started=False, actual_first1000_checkpoint_verified=False,
        failed_original_joint_progress_preserved_and_not_credited=progress['optimizer_updates'],
        failed_training_worker_verified_not_live=old_worker,
        previous_replay_identity=dict(pid=actor.pid, create_time=actor.create_time(), command=command),
        previous_replay_descendants=descendants, previous_replay_private_bytes=actor.memory_info().private,
        controller_only_idle_handoff_registered=True, original_guards_and_seed_slots_unchanged=True,
        actual_fullnative_checkpoint_api_validation_pending_training=True,
        synthetic_or_mock_persistence_tests_are_not_benchmark_or_native_training_evidence=True,
        all_original_experiments_complete=False)
    exclusive(RESULT+'/actual_full50000_checkpointed_recovery_registration.json',record)
    return record


if __name__ == '__main__':
    print(register())
