"""Freeze successful native predecessor stages and the remaining full evaluations."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, verify_queue
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    original_path = 'configs/experiments/fm_timegan_native_original_queue.v1.json'
    queue = read(ROOT / original_path); verify_queue(queue)
    job = next(j for j in queue['jobs'] if j['id'] == 'timegan_native__gru__stock__seed1104')
    previous = ROOT / job['output_directory']
    path = 'configs/reproducibility/fm_timegan_stock1104_posthoc_recovery.2026-10-12_v1.json'
    terminal = queue['state_root'] + '/receipts/' + job['id'] + '.json'
    failure = queue['state_root'] + '/logs/' + job['id'] + '.stderr.log'
    sources = {p['path'] for p in queue['source_receipts']}
    sources.update([original_path, 'scripts/flow_matching/timegan_posthoc_recovery_v1.py',
        'scripts/flow_matching/run_timegan_posthoc_recovery_v1.py',
        'scripts/flow_matching/register_timegan_posthoc_recovery_v1.py',
        'scripts/flow_matching/run_timegan_full45_completion_replay_v2.py',
        'scripts/flow_matching/run_timegan_full45_completion_replay_v1.py',
        'scripts/flow_matching/timegan_stage_resume_v2.py',
        'scripts/flow_matching/timegan_predictive_observer_v2.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'tests/test_fm_timegan_posthoc_recovery_v1.py'])
    tags = ['embedding_complete', 'supervised_complete', 'timegan_complete'] + [
        'discriminative_%02d_complete' % i for i in range(10)]
    archives = ['full_native_generation_feed.npz', 'full_native_real_and_generated.npz'] + [
        'discriminative_%02d_full_test.npz' % i for i in range(10)]
    config = dict(original_queue=original_path, original_queue_sha256=sha(ROOT / original_path), canonical_job=job,
        output_directory='F:/aicoding/IIA_Data/experiments/timegan_posthoc_recovery_v1/jobs/' + job['id'],
        state_root='experiments/runs/fm_timegan_stock1104_posthoc_recovery_v1',
        terminal_receipt=dict(path=terminal, sha256=sha(ROOT / terminal)),
        failure_log=dict(path=failure, sha256=sha(ROOT / failure)),
        checkpoint_receipts=[dict(tag=tag, sha256=sha(previous / 'checkpoints' / tag / 'receipt.json')) for tag in tags],
        archive_receipts=[dict(name=name, sha256=sha(previous / name)) for name in archives],
        completed_evaluators_sha256=sha(previous / 'completed_evaluators.json'),
        settings=copy.deepcopy(queue['settings']), resource_lock=queue['resource_lock'],
        python=queue['python'], cpu_threads=queue['cpu_threads'],
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        citation='Yoon, Jarrett, van der Schaar. Time-series Generative Adversarial Networks. NeurIPS 2019.',
        paper='https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks',
        code='https://github.com/jsyoon0823/TimeGAN',
        boundary='All successful native generator and ten discriminative evaluator checkpoints remain attributed to the '
            'preserved failed observer attempt. All ten original predictors must execute 5000 actual optimizer updates '
            'each, from the exact discriminator09 post-evaluator NumPy state and original graph seeds. Unsaved failed '
            'predictor updates are discarded, not credited. All23 checkpoint/model/optimizer restores and all20 full '
            'evaluator replays still require independent bitwise certification; registration is not experiment completion.',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)])
    from scripts.flow_matching.timegan_posthoc_recovery_v1 import binding, compile_recovery
    binding(config); compile_recovery(config)
    exclusive(ROOT / path, config)
    replay_path = 'configs/runtime/fm_timegan_full45_completion_replay.v2.json'
    old_path = 'configs/runtime/fm_timegan_full45_completion_replay.v1.json'
    old = read(ROOT / old_path)
    state = read(ROOT / old['state_root'] / 'status.json')
    process = psutil.Process(state['pid'])
    expected = dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline())
    from scripts.flow_matching.run_timegan_full45_completion_replay_v2 import handoff_phase_allowed, runtime_namespace
    children = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]
    if (state['create_time'] != expected['create_time']
            or 'scripts.flow_matching.run_timegan_full45_completion_replay_v1' not in expected['command']
            or not handoff_phase_allowed('original', state, False, children)):
        raise ValueError('Only exact idle full45 owner without children or completion history can hand off')
    replay = copy.deepcopy(old)
    replay.update(runtime_configuration=replay_path,
        state_root='experiments/runs/fm_timegan_full45_completion_replay_v2',
        output_root='projects/flow_matching_research/results/2026-10-12/timegan_full45_completion_replay_v2/completed_models',
        previous_replay_owner=expected, previous_replay_state_path=old['state_root'] + '/status.json')
    route = next(r for r in replay['routes'] if r['canonical_job']['id'] == job['id'])
    route.update(kind='posthoc_recovery', execution_job=dict(job, output_directory=config['output_directory']),
        result_path=config['output_directory'] + '/result.json', recovery_configuration=path,
        native_completion_receipts=[dict(path=config['state_root'] + '/receipt.json',
            successful_field='full_native_posthoc_recovery_verified')])
    replay['boundary'] = config['boundary'] + ' All45 original slots, bitwise replay loops and unchanged caps retained.'
    replay_sources = {p['path'] for p in old['source_receipts']} | sources | {path, old_path}
    replay['source_receipts'] = [dict(path=p, sha256=sha(ROOT / p)) for p in sorted(replay_sources)]
    runtime_namespace()['verify'](replay)
    exclusive(ROOT / replay_path, replay)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_stock1104_posthoc_recovery_v1'
    exclusive(evidence / 'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path, configuration_sha256=sha(ROOT / path), full45_replay_configuration=replay_path,
        full45_replay_configuration_sha256=sha(ROOT / replay_path), canonical_job=job,
        original_scope=config['original_scope'], reused_complete_generator_checkpoints=3,
        reused_complete_discriminative_checkpoints=10, fresh_predictive_training_updates_required=50000,
        independent_full_checkpoint_restores_required=23, independent_trained_evaluator_replays_required=20,
        training_seed_slots_added=0, author_algorithm_changed=False, original_guards_unchanged=True,
        old_replay_idle_owner=expected, old_replay_native_children=children,
        native_checkpoint_reinference_performed=False, new_benchmark_results=0,
        experiment_complete=False, all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path, replay_configuration=replay_path, full_original_TimeGAN_jobs=45)))


if __name__ == '__main__':
    main()
