"""Retry two authoritative failed NODE slots without changing the original study."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, read, sha, verify, binding
from scripts.flow_matching.run_light_controller import load_functions
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

load_functions(ROOT, 'scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py',
               ['validate_recovery'], globals())


def main():
    original_path = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
    queue_path = 'configs/experiments/fm_cfm_ts_exact_recovery_queue.v7.json'
    runtime_path = 'configs/runtime/fm_cfm_ts_low_memory_controller.v11.json'
    evidence_root = ROOT / 'projects/flow_matching_research/results/2026-10-11/cfm_node_original_resource_retry_v6'
    for path in (ROOT / queue_path, ROOT / runtime_path, evidence_root):
        if path.exists():
            raise FileExistsError('Keep all preceding CFM attempts and registrations: ' + str(path))
    original = read(ROOT / original_path)
    verify(original)
    queue = deepcopy(original)
    jobs = {j['id']:j for j in queue['jobs']}
    cases = [
        ('cfm_ts_original__main_text__node_e400__pendulum__seed1103',
         'configs/experiments/fm_cfm_ts_exact_recovery_queue.v5.json',
         'cfm_ts_original__main_text__node_e400__pendulum__seed1103__exact_recovery_v4'),
        ('cfm_ts_original__main_text__node_e400__pendulum__seed1104', original_path,
         'cfm_ts_original__main_text__node_e400__pendulum__seed1104'),
    ]
    observed = []
    for handle in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            command = handle.info['cmdline'] or []
            if any('run_cfm_ts_original_job' in arg for arg in command):
                observed.append(handle.info)
        except psutil.Error:
            continue
    evidence, selected, source_paths = [], [], []
    for canonical, prior_path, prior_id in cases:
        prior = read(ROOT / prior_path)
        verify(prior)
        prior_job = next(j for j in prior['jobs'] if j['id']==prior_id)
        if any(any(canonical in arg for arg in p['cmdline']) for p in observed):
            raise RuntimeError('An original or recovered worker for this seed is live: ' + canonical)
        if (ROOT / jobs[canonical]['output_directory'] / 'result.json').exists() or (
                ROOT / prior_job['output_directory'] / 'result.json').exists():
            raise ValueError('Review existing full results before retrying: ' + canonical)
        receipt_path = prior['state_root'] + '/receipts/' + prior_id + '.json'
        receipt = read(ROOT / receipt_path)
        if receipt['complete'] or receipt['exit_code']==0 or not receipt['resource_observations']['resource_guard_aborted']:
            raise ValueError('An authoritative resource-aborted full-budget job is required')
        # Scan every prior alias so an already completed recovery is not counted twice.
        for previous_path in sorted((ROOT/'configs/experiments').glob('fm_cfm_ts_exact_recovery_queue.*.json')):
            previous = read(previous_path)
            for candidate in previous['jobs']:
                if candidate.get('recovery_original_job_id')==canonical and (
                        ROOT/candidate['output_directory']/'result.json').exists():
                    raise ValueError('An earlier recovery result needs completion review before retrying')
        progress_path = prior_job['output_directory'] + '/progress.json'
        failure = dict(canonical_id=canonical, prior_job_id=prior_id, prior_queue=prior_path,
            prior_queue_sha256=sha(ROOT/prior_path), terminal_receipt=receipt,
            terminal_receipt_path=receipt_path,terminal_receipt_sha256=sha(ROOT/receipt_path),
            prior_output_preserved=prior_job['output_directory'], original_output_preserved=jobs[canonical]['output_directory'],
            no_matching_live_worker_observed=True, prior_native_workers=observed,
            checkpoint_resume_claimed=False, prior_partial_updates_not_counted_as_completed_training=True)
        source_paths += [prior_path,receipt_path]
        if (ROOT/progress_path).exists():
            failure.update(preserved_progress=read(ROOT/progress_path), preserved_progress_path=progress_path,
                           preserved_progress_sha256=sha(ROOT/progress_path))
            source_paths.append(progress_path)
        evidence.append(failure)
        jobs[canonical].update(id=canonical+'__exact_recovery_v6',recovery_original_job_id=canonical,
            output_directory='experiments/runs/fm_cfm_ts_exact_recovery_v6/jobs/'+canonical+'__exact_recovery_v6',
            artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_ts_exact_recovery_v6/'+canonical+'__exact_recovery_v6')
        selected.append(jobs[canonical]['id'])
    source_paths += [original_path,Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/flow_matching/register_grasp_tau0_seed1104_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_receipt_runtime_v3.py']
    extra = [dict(path=p,sha256=sha(ROOT/p)) for p in dict.fromkeys(source_paths)]
    queue.update(state_root='experiments/runs/fm_cfm_ts_exact_recovery_v6',
        recovery_original_queue=original_path,recovery_original_queue_sha256=sha(ROOT/original_path),
        recovery_job_ids=selected,recovery_evidence=evidence,registered_utc=datetime.now(timezone.utc).isoformat(),
        recovery_selection='First full verified exact-budget attempt resolves the same canonical seed; no metric selection or extra seed.',
        additional_training_seed_slots=0)
    queue['source_receipts'] += extra
    validate_recovery(original,queue)
    predecessor_path='configs/runtime/fm_cfm_ts_low_memory_controller.v10.json'
    predecessor=read(ROOT/predecessor_path)
    binding(predecessor,'recovery')
    exclusive_json(ROOT/queue_path,queue)
    runtime=dict(controllers=deepcopy(predecessor['controllers']), family_lock=predecessor['family_lock'],
        source_receipts=predecessor['source_receipts']+extra+[dict(path=predecessor_path,sha256=sha(ROOT/predecessor_path))],
        boundary='Original285canonical slots preserved. Retry only two authoritative resource-aborted NODE '
          'main-text pendulum seeds1103/1104, full400epochs, batch256, full data and original adaptiveODE '
          'settings unchanged. Previous partials, completed recoveries and live seed1105/1106 queues retained. '
          'Original shared CFM family lock and resource safeguards remain. No extra independent seed or benchmark result.')
    runtime['controllers']['recovery']=dict(queue=queue_path,queue_sha256=sha(ROOT/queue_path),selected_job_ids=selected)
    binding(runtime,'recovery')
    exclusive_json(ROOT/runtime_path,runtime)
    exclusive_json(evidence_root/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        original_canonical_jobs=len(original['jobs']), selected_full_budget_retries=selected,
        prior_failed_attempts=evidence,queue=queue_path,runtime=runtime_path,
        source_receipts=extra,additional_training_seed_slots=0,full_training_and_evaluation_completed=False,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        all_original_experiments_complete=False,boundary=runtime['boundary']))
    print(json.dumps(dict(original_slots=285,full_budget_retry_jobs=len(selected),selected=selected)))


if __name__=='__main__':
    main()
