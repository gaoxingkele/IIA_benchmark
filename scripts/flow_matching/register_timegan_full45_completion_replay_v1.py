"""Register all45 original TimeGAN models and complete trained critic replays."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import ROOT, read, sha, verify, exclusive


def main():
    path = 'configs/runtime/fm_timegan_full45_completion_replay.v1.json'
    original_path = 'configs/experiments/fm_timegan_native_original_queue.v1.json'
    queue = read(ROOT / original_path)
    corrected_path = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v2.json'
    corrected = read(ROOT / corrected_path)
    model = read(ROOT / queue['model_config'])
    datasets = read(ROOT / model['dataset_config'])['datasets']
    sources = {original_path, corrected_path,
        'scripts/flow_matching/run_timegan_full45_completion_replay_v1.py',
        'scripts/flow_matching/register_timegan_full45_completion_replay_v1.py',
        'scripts/flow_matching/run_timegan_native_queue_v1.py',
        'scripts/flow_matching/timegan_stage_resume_v2.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'tests/test_fm_timegan_full45_completion_replay_v1.py'}
    sources.update(pin['path'] for pin in queue['source_receipts'])
    sources.update(pin['path'] for pin in corrected['source_receipts'])
    routes = []
    recoveries = {corrected['roles'][role]['canonical_job_id']: role for role in ('stock', 'energy')}
    for canonical in queue['jobs']:
        if canonical['id'] in recoveries:
            role = corrected['roles'][recoveries[canonical['id']]]
            recovery_path = role['recovery_configuration']
            recovery = read(ROOT / recovery_path)
            execution = dict(canonical, output_directory=recovery['output_directory'])
            sources.add(recovery_path)
            sources.update(pin['path'] for pin in recovery.get('recovery_source_receipts', []))
            completion_receipts = [dict(path=role['state_root'] + '/receipts/' + canonical['id'] + '.json',
                successful_field='full_native_result_verified')]
            kind = 'stage_resume'
        else:
            execution = copy.deepcopy(canonical)
            recovery_path = None; kind = 'original'
            completion_receipts = [dict(path=corrected['roles']['original']['state_root'] + '/receipts/' + canonical['id'] + '.json',
                successful_field='full_native_result_verified'),
                dict(path=queue['state_root'] + '/receipts/' + canonical['id'] + '.json', successful_field='full_native_result_verified')]
        routes.append(dict(canonical_job=copy.deepcopy(canonical), execution_job=execution, kind=kind,
            full_original_data_shape=datasets[canonical['dataset']]['full_shape'],
            recovery_configuration=recovery_path, native_completion_receipts=completion_receipts,
            result_path=(Path(execution['output_directory']) / 'result.json').as_posix()))
    owners = []
    for role in ('original', 'stock', 'energy'):
        state = read(ROOT / corrected['roles'][role]['state_root'] / 'status.json')
        process = psutil.Process(state['pid'])
        command = process.cmdline()
        if (process.create_time() != state['create_time']
                or 'scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2' not in command
                or command[-1] != role):
            raise ValueError('Actual original corrected training controller identity changed')
        owners.append(dict(pid=process.pid, create_time=process.create_time(), command=command))
    for pid, created, fragment in [(32676, 1791669201.5630853, 'run_timegan_native_queue_v1'),
        (33752, 1791740029.9719155, 'run_timegan_native_job_v1')]:
        process = psutil.Process(pid)
        if process.create_time() != created or not any(fragment in part for part in process.cmdline()):
            raise ValueError('Actual original active native training identity changed')
        owners.append(dict(pid=pid, create_time=created, command=process.cmdline()))
    generated_windows = sum(r['full_original_data_shape'][0] for r in routes)
    config = dict(runtime_configuration=path, original_queue=original_path, routes=routes,
        required_checkpoints_each=23, required_evaluators_each=20, equality_gate='bitwise',
        protocol_iterations_each=70000, required_full_generated_windows=generated_windows,
        required_full_predictive_test_windows=10*generated_windows,
        required_full_discriminative_test_windows=sum(20 * (r['full_original_data_shape'][0] - int(r['full_original_data_shape'][0] * .8)) for r in routes),
        cpu_threads=queue['cpu_threads'], python=queue['python'], settings=copy.deepcopy(queue['settings']),
        resource_lock=queue['resource_lock'], poll_seconds=10,
        state_root='experiments/runs/fm_timegan_full45_completion_replay_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/timegan_full45_completion_replay_v1/completed_models',
        expected_original_native_owners=owners,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        citation='Yoon, Jarrett, van der Schaar. Time-series Generative Adversarial Networks. NeurIPS 2019.',
        paper='https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks',
        code='https://github.com/jsyoon0823/TimeGAN',
        boundary='All45 full native models and900 complete trained evaluators remain obligations. Suppressed numerical '
            'optimizer fetches are used only to reconstruct the untouched author sampling protocol from saved RNG; '
            'they are never counted as training. All complete trained checkpoint predictions and scores must be bitwise '
            'equal. Actual successful full native training is a prerequisite. Original overlapping-window/global '
            'normalization protocol is not TAB. Current unpatched active stock1104 requires a separate exact recovery '
            'mapping if its frozen observer fails. No complete result or model equivalence is claimed by registration.',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)])
    verify(config)
    exclusive(ROOT / path, config)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_full45_completion_replay_v1'
    exclusive(evidence / 'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path, configuration_sha256=sha(ROOT / path), original_scope=config['original_scope'],
        original_TimeGAN_jobs=45, required_full_checkpoint_restores=1035,
        required_trained_evaluator_replays=900, required_complete_generated_windows=generated_windows,
        required_full_predictive_test_windows=config['required_full_predictive_test_windows'],
        required_full_discriminative_test_windows=config['required_full_discriminative_test_windows'],
        required_original_sampling_protocol_iterations=3150000,
        numerical_optimizer_updates_to_execute=0, additional_training_seed_slots=0,
        original_native_training_controllers_unchanged=True, full_native_replays_executed=False,
        actual_original_native_owner_identities=owners, new_numeric_results_published=0,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path, original_jobs=45, required_checkpoints=1035,
        required_trained_evaluator_replays=900, full_generated_windows=generated_windows,
        source_pins=len(config['source_receipts']))))


if __name__ == '__main__':
    main()
