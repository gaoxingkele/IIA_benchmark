"""Pin the observational correction and exact three native owner handoffs."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching.timegan_predictive_observer_v2 import ROOT, read, sha, verify_runtime
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import exclusive


def main():
    path = 'configs/runtime/fm_timegan_corrected_observer_dispatch.v2.json'
    queue_path = 'configs/experiments/fm_timegan_native_original_queue.v1.json'
    queue = read(ROOT / queue_path)
    definitions = {
        'original': (32676, 1791669201.5630853, 'run_timegan_native_queue_v1',
            queue['state_root'] + '/status.json', None),
        'stock': (35356, 1791720569.0788147, 'run_timegan_stage_resume_v2',
            'experiments/runs/fm_timegan_stock_stage_recovery_v3/status.json',
            'configs/reproducibility/fm_timegan_stock_stage_recovery.2026-10-11_v3.json'),
        'energy': (16104, 1791740868.216404, 'run_preserved_native_gan_retry_v1',
            'experiments/runs/fm_timegan_preserved_full_budget_retry_v1/status.json',
            'configs/reproducibility/fm_timegan_energy_preserved_full_joint_retry.2026-10-12_v1.json'),
    }
    roles = {}
    sources = {queue_path, 'scripts/flow_matching/timegan_predictive_observer_v2.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
        'scripts/flow_matching/register_timegan_corrected_observer_dispatch_v2.py',
        'scripts/flow_matching/timegan_stage_resume_v2.py',
        'scripts/flow_matching/run_timegan_stage_resume_v2.py',
        'scripts/flow_matching/run_timegan_native_queue_v1.py',
        'scripts/flow_matching/run_preserved_native_gan_retry_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'tests/test_fm_timegan_predictive_observer_v2.py'}
    sources.update(pin['path'] for pin in queue['source_receipts'])
    for role, (pid, created, fragment, state_path, recovery_path) in definitions.items():
        owner = psutil.Process(pid)
        if owner.create_time() != created or not any(fragment in part for part in owner.cmdline()):
            raise ValueError('Actual old controller identity changed; inspect before registration')
        state = read(ROOT / state_path)
        if state.get('pid') != pid or state.get('create_time') != created:
            raise ValueError('Old native owner and its authoritative state disagree')
        settings = copy.deepcopy(queue['settings'])
        route = dict(runtime_configuration=path, previous_owner=dict(pid=pid, create_time=created, command=owner.cmdline()),
            previous_state_path=state_path, state_root='experiments/runs/fm_timegan_corrected_observer_' + role + '_v2',
            recovery_configuration=recovery_path, settings=settings)
        if recovery_path:
            recovery = read(ROOT / recovery_path)
            if (ROOT / recovery['output_directory']).exists() or state.get('active_pid') is not None:
                raise ValueError('Only an unstarted recovery can be handed off for observation correction')
            route.update(output_directory=recovery['output_directory'], canonical_job_id=recovery['canonical_job_id'],
                canonical_job=recovery['canonical_job'])
            if 'retry_admission_commit_headroom_bytes' in recovery:
                from scripts.flow_matching.run_preserved_native_gan_retry_v1 import admission_settings
                route['settings'] = admission_settings(recovery, settings)
            sources.add(recovery_path)
            sources.update(pin['path'] for pin in recovery.get('recovery_source_receipts', []))
        roles[role] = route
    config = dict(runtime_configuration=path, original_queue=queue_path, roles=roles,
        correction='predictive_observation_only', cpu_threads=queue['cpu_threads'],
        resource_lock=queue['resource_lock'], poll_seconds=10,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        citation='Yoon, Jarrett, van der Schaar. Time-series Generative Adversarial Networks. NeurIPS 2019.',
        paper='https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks',
        author_source='https://github.com/jsyoon0823/TimeGAN',
        boundary='All45 original jobs/seeds/budgets remain obligations. Only predictive NPZ target observation changes. '
            'Active native training is never stopped or edited. Current unpatched active job may still encounter its '
            'frozen recording bug; its full checkpoints and failed attempt require a separate exact continuation. '
            'No new numeric result, model equivalence, TAB certification or all-experiment completion claimed.',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)])
    verify_runtime(config)
    exclusive(ROOT / path, config)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_predictive_observer_v2'
    exclusive(evidence / 'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        runtime_configuration=path, configuration_sha256=sha(ROOT / path), original_scope=config['original_scope'],
        full_original_TimeGAN_jobs=45, full_stages_per_seed=[50000, 50000, 50000],
        full_evaluator_repeats_per_seed=dict(discriminative=10, predictive=10),
        full_evaluator_updates_per_repeat=dict(discriminative=2000, predictive=5000),
        original_native_owners={k: v['previous_owner'] for k, v in roles.items()},
        source_observation_expression_changes=1, author_training_and_scoring_sources_unchanged=True,
        resource_admission_and_emergency_caps_unchanged=True, active_native_workers_stopped=False,
        training_executed_during_registration=False, new_numeric_results_published=0,
        original_active_unpatched_job_requires_separate_exact_continuation_if_it_fails=True,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path, full_original_jobs=45, source_pins=len(config['source_receipts']))))


if __name__ == '__main__':
    main()
