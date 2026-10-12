"""Admit full Sine recovery only after actual native restore proof succeeds."""
import copy
from datetime import datetime, timezone
import json

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    audit_runtime_path = 'configs/runtime/fm_timegan_sine_stage_restore_audit.v3.json'
    runtime = read(ROOT / audit_runtime_path)
    from scripts.flow_matching.run_timegan_sine_stage_audit_v3 import verify
    config, queue, job, stages = verify(runtime)
    receipt_path = runtime['state_root'] + '/receipt.json'
    receipt = read(ROOT / receipt_path)
    proof_path = config['native_restore_audit']; proof = read(ROOT / proof_path)
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] is not False
            or receipt['full_native_restore_audit_passed'] is not True
            or receipt['audit_sha256'] != sha(ROOT / proof_path)
            or proof['passed'] is not True or proof['all_transition_variable_values_bitwise_equal'] is not True
            or proof['numpy_minibatch_noise_trajectory_identical'] is not True
            or proof['full_original_data_shape'] != [10000,24,5]
            or proof['checkpoint_all_variable_restoration']['all_variable_values_bitwise_equal'] is not True):
        raise ValueError('Actual successful complete native Sine restore proof required')
    path = 'configs/reproducibility/fm_timegan_sine_stage_recovery.2026-10-12_v3.json'
    sources = {p['path'] for p in runtime['source_receipts']}
    sources.update([audit_runtime_path, proof_path, receipt_path,
        'scripts/flow_matching/run_timegan_corrected_stage_recovery_v3.py',
        'scripts/flow_matching/register_timegan_sine_stage_recovery_v3.py',
        'scripts/flow_matching/run_timegan_stage_resume_v2.py',
        'tests/test_fm_timegan_corrected_stage_recovery_v3.py'])
    recovery = dict(config, native_restore_audit_sha256=sha(ROOT / proof_path),
        original_training_settings=copy.deepcopy(queue['settings']),
        recovery_source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)])
    from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_resume_compiler
    corrected_resume_compiler()(recovery)
    exclusive(ROOT / path, recovery)
    old_path = 'configs/runtime/fm_timegan_full45_completion_replay.v2.json'
    old = read(ROOT / old_path)
    state_path = old['state_root'] + '/status.json'; state = read(ROOT / state_path)
    process = psutil.Process(state['pid'])
    command = process.cmdline()
    from scripts.flow_matching.run_timegan_full45_completion_replay_v2 import handoff_phase_allowed, runtime_namespace
    children = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]
    if (process.create_time() != state['create_time']
            or 'scripts.flow_matching.run_timegan_full45_completion_replay_v2' not in command
            or command[-1] != old_path or not handoff_phase_allowed('original', state, False, children)):
        raise ValueError('Exact idle full45 owner without any native child or completion history required')
    replay_path = 'configs/runtime/fm_timegan_full45_completion_replay.v3.json'
    replay = copy.deepcopy(old)
    replay.update(runtime_configuration=replay_path, state_root='experiments/runs/fm_timegan_full45_completion_replay_v3',
        output_root='projects/flow_matching_research/results/2026-10-12/timegan_full45_completion_replay_v3/completed_models',
        previous_replay_owner=dict(pid=process.pid, create_time=process.create_time(), command=command),
        previous_replay_state_path=state_path)
    route = next(r for r in replay['routes'] if r['canonical_job']['id'] == job['id'])
    route.update(kind='stage_resume', execution_job=dict(job, output_directory=recovery['output_directory']),
        recovery_configuration=path, result_path=recovery['output_directory'] + '/result.json',
        native_completion_receipts=[dict(path=recovery['state_root'] + '/receipt.json',
            successful_field='full_native_recovery_result_verified')])
    sources.update(p['path'] for p in old['source_receipts'])
    sources.update([path, old_path])
    replay['source_receipts'] = [dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)]
    replay['boundary'] += ' Sine1103 now reuses its two actual verified full pretraining stages instead of '
    replay['boundary'] += 'an unstarted redundant fresh retry; full joint/all20 evaluators and all23 full bitwise replays remain.'
    runtime_namespace()['verify'](replay)
    exclusive(ROOT / replay_path, replay)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_sine_stage_restore_audit_v3'
    exclusive(evidence / 'full_training_recovery_registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path, configuration_sha256=sha(ROOT / path), replay_configuration=replay_path,
        replay_configuration_sha256=sha(ROOT / replay_path), original_scope=runtime['original_scope'],
        original_TimeGAN_jobs_retained=45, unchanged_other_native_replay_routes=44,
        verified_full_native_variables_restored=proof['checkpoint_all_variable_restoration']['restored_native_variables'],
        verified_native_restore_proof_sha256=sha(ROOT / proof_path),
        preserved_pretraining_updates=stages[-1]['optimizer_updates'],
        remaining_full_native_joint_iterations=50000, remaining_full_native_evaluator_trainings=20,
        old_idle_replay_owner=replay['previous_replay_owner'], original_training_guards_unchanged=True,
        discarded_unsaved_joint_updates_not_counted=True, additional_training_seed_slots=0,
        full_training_recovery_complete=False, full_checkpoint_replay_complete=False,
        new_benchmark_results=0, all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path, replay_configuration=replay_path, variables_restored=284,
        full_original_jobs=45, other_routes_unchanged=44)))


if __name__ == '__main__':
    main()
