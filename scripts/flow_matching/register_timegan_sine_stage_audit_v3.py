"""Register a native full checkpoint audit instead of repeating saved pretraining."""
import copy
from datetime import datetime, timezone
import json

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.timegan_stage_resume_v2 import binding
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    import psutil
    previous_path = 'configs/runtime/fm_timegan_shared_commit_full_retry.v1.json'
    previous = read(ROOT / previous_path)
    original_path = previous['original_queue']; queue = read(ROOT / original_path)
    job = next(j for j in queue['jobs'] if j['id'] == previous['selected_job_id'])
    stage_directory = ROOT / job['output_directory'] / 'checkpoints'
    audit_path = 'configs/reproducibility/fm_timegan_sine_stage_restore_audit.2026-10-12_v3.json'
    runtime_path = 'configs/runtime/fm_timegan_sine_stage_restore_audit.v3.json'
    audit_config = dict(original_queue=original_path, original_queue_sha256=sha(ROOT / original_path),
        canonical_job_id=job['id'], canonical_job=job,
        stage_receipt_sha256={tag:sha(stage_directory / tag / 'receipt.json') for tag in ['embedding_complete','supervised_complete']},
        output_directory='F:/aicoding/IIA_Data/experiments/timegan_sine_stage_recovery_v3/jobs/' + job['id'],
        state_root='experiments/runs/fm_timegan_sine_stage_recovery_v3',
        native_restore_audit='projects/flow_matching_research/results/2026-10-12/timegan_sine_stage_restore_audit_v3/full_native_restore_and_transition_audit.json',
        audit_joint_iterations=2, independent_training_seed_slots_added=0, full_scope_original_jobs=45,
        citation='Yoon, Jarrett, van der Schaar. Time-series Generative Adversarial Networks. NeurIPS 2019.',
        paper='https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks',
        code='https://github.com/jsyoon0823/TimeGAN', all_original_experiments_complete=False,
        boundary='Both50000-update native pretraining stages already completed and remain preserved. Full original '
            'Sine10000x24x5 loader/model/checkpoint/optimizer/RNG restoration and exact native transitions must pass '
            'before recovery training is admitted. Transition audit is not benchmark training/performance. Entire '
            '50000-iteration original joint loop and all20 full evaluators remain required. Unsaved interrupted '
            'joint updates are discarded, never credited. Original45TimeGAN slots and45paper681method273task scope retained.')
    binding(audit_config)
    exclusive(ROOT / audit_path, audit_config)
    old_state_path = previous['state_root'] + '/status.json'
    state = read(ROOT / old_state_path); process = psutil.Process(state['pid'])
    command = process.cmdline()
    from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import handoff_phase_allowed
    children = [dict(pid=p.pid, name=p.name(), exe=p.exe()) for p in process.children(recursive=True)]
    if (process.create_time() != state['create_time']
            or 'scripts.flow_matching.run_selected_native_full_retry_v1' not in command
            or command[-1] != previous_path or not handoff_phase_allowed('original', state, False, children)
            or (ROOT / previous['selected_output_directory']).exists()):
        raise ValueError('Only the exact unstarted memory-waiting selected retry may be handed off')
    reference_path = 'configs/runtime/fm_timegan_energy_stage_restore_audit.v1.json'
    reference = read(ROOT / reference_path)
    sources = {p['path'] for p in reference['source_receipts']} | {p['path'] for p in previous['source_receipts']}
    sources.update([previous_path, audit_path, reference_path,
        'scripts/flow_matching/audit_timegan_sine_stage_restore_v3.py',
        'scripts/flow_matching/run_timegan_sine_stage_audit_v3.py',
        'scripts/flow_matching/register_timegan_sine_stage_audit_v3.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
        'scripts/flow_matching/timegan_predictive_observer_v2.py',
        'tests/test_fm_timegan_sine_stage_audit_v3.py'])
    runtime = dict(audit_configuration=audit_path, previous_registration=previous_path,
        reference_audit_runtime=reference_path, python=queue['python'], settings=copy.deepcopy(reference['settings']),
        resource_lock=reference['resource_lock'], state_root='experiments/runs/fm_timegan_sine_stage_restore_audit_v3',
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        handoff=dict(previous_owner=dict(pid=process.pid, create_time=process.create_time(), command=command),
            previous_state_path=old_state_path, runtime_configuration=runtime_path),
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)],
        boundary=audit_config['boundary'])
    from scripts.flow_matching.run_timegan_sine_stage_audit_v3 import verify
    verify(runtime)
    exclusive(ROOT / runtime_path, runtime)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-12/timegan_sine_stage_restore_audit_v3'
    exclusive(evidence / 'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        audit_configuration=audit_path, audit_configuration_sha256=sha(ROOT / audit_path),
        runtime_configuration=runtime_path, runtime_configuration_sha256=sha(ROOT / runtime_path),
        canonical_job=job, original_scope=runtime['original_scope'], full_original_data_shape=[10000,24,5],
        preserved_stage_updates=dict(E0=50000,GS=50000,E=0,G=0,D=0),
        two_transition_iterations_per_graph_are_verification_only=True,
        actual_previous_waiting_owner=runtime['handoff']['previous_owner'], actual_previous_children=children,
        original_audit_guards_unchanged=True, native_restore_audit_complete=False,
        full_training_recovery_registered=False, full45_completion_route_updated=False,
        new_benchmark_results=0, all_original_experiments_complete=False))
    print(json.dumps(dict(runtime_configuration=runtime_path, original_jobs=45, saved_pretraining_updates=100000)))


if __name__ == '__main__':
    main()
