"""Preserve an authoritative failed attempt and retry the same original seed."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.timegan_stage_resume_v2 import binding
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha


def main():
    prior_path = 'configs/reproducibility/fm_timegan_stock_stage_recovery.2026-10-11_v2.json'
    target_path = 'configs/reproducibility/fm_timegan_stock_stage_recovery.2026-10-11_v3.json'
    evidence_root = ROOT / 'projects/flow_matching_research/results/2026-10-11/timegan_stock_stage_retry_v3'
    if (ROOT / target_path).exists() or evidence_root.exists():
        raise FileExistsError('Preserve every prior registered recovery')
    previous = read(ROOT / prior_path)
    state_path = previous['state_root'] + '/status.json'
    receipt_path = previous['state_root'] + '/receipt.json'
    state, receipt = read(ROOT / state_path), read(ROOT / receipt_path)
    if (state['state'] != 'failed_attempt_preserved' or receipt['exit_code'] == 0
            or receipt['full_native_recovery_result_verified'] or not receipt['resource_guard_aborted']
            or state['active_pid'] is not None):
        raise ValueError('An authoritative resource-aborted terminal attempt is required')
    matching = []
    for process in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if any('run_timegan_stage_resume_v2' in arg for arg in command) and any(prior_path in arg for arg in command):
                matching.append(process.info)
        except psutil.Error:
            continue
    if matching:
        raise RuntimeError('The previous registered attempt still has a live process')
    progress_path = previous['output_directory'] + '/progress.json'
    progress = read(ROOT / progress_path)
    if progress['full_original_training_complete'] or progress['full_evaluation_complete']:
        raise ValueError('Inspect unexpected full training evidence before retrying')
    config = deepcopy(previous)
    config['output_directory'] = previous['output_directory'].replace('_v2/', '_v3/')
    config['state_root'] = 'experiments/runs/fm_timegan_stock_stage_recovery_v3'
    failure_sources = [prior_path, receipt_path, state_path, progress_path,
                       previous['state_root'] + '/worker.stdout.log', previous['state_root'] + '/worker.stderr.log',
                       Path(__file__).relative_to(ROOT).as_posix(),
                       'scripts/flow_matching/register_grasp_tau0_seed1104_replay_v1.py']
    config['recovery_source_receipts'] += [dict(path=p, sha256=sha(ROOT / p)) for p in failure_sources]
    config['prior_terminal_attempt'] = dict(configuration=prior_path, receipt_path=receipt_path,
        receipt_sha256=sha(ROOT / receipt_path), original_output_preserved=previous['output_directory'],
        captured_progress_optimizer_updates=progress['optimizer_updates'], resource_guard_aborted=True,
        no_previous_live_process_observed=True, unsaved_joint_progress_reused=False)
    config['boundary'] = (previous['boundary'] + ' Retry v3 preserves the failed v2 attempt, '
        'its resource-abort receipt and all unsaved joint progress. Restore the same verified '
        'embedding/supervised full-stage checkpoints and execute all50000joint iterations '
        'and20full evaluator trainings anew; no fabricated continuation from unsaved44000joint iterations. '
        'Original family lock, CPU settings and memory safeguards unchanged.')
    queue, model, job, stages = binding(config)
    original_queue, original_model, original_job, original_stages = binding(previous)
    if (queue != original_queue or model != original_model or job != original_job or stages != original_stages
            or config['output_directory'] == previous['output_directory']):
        raise ValueError('Original scientific recipe and complete-stage restoration must remain unchanged')
    exclusive_json(ROOT / target_path, config)
    exclusive_json(evidence_root / 'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        canonical_job_id=job['id'], original_queue_jobs=len(queue['jobs']), recovery_configuration=target_path,
        recovery_configuration_sha256=sha(ROOT / target_path), terminal_receipt=receipt,
        failed_attempt_status=state, failed_attempt_progress=progress,
        status_resource_flag_disagrees_with_terminal_receipt=state.get('resource_guard_aborted') != receipt['resource_guard_aborted'],
        authoritative_failure_source='terminal_resource_receipt', prior_complete_stages_restored=2,
        independent_training_seed_slots_added=0, full_training_and_evaluation_completed=False,
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        all_original_experiments_complete=False, source_receipts=config['recovery_source_receipts'],
        boundary=config['boundary']))
    print(json.dumps(dict(recovery_configuration=target_path, same_original_seed=job['seed'],
                          exact_budget_preserved=True, full_training_completed=False)))


if __name__ == '__main__':
    main()
