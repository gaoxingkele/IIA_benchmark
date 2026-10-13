"""Preserve the terminal 99-epoch attempt and register the same full CFM seed."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, read, sha, binding, verify
from scripts.flow_matching.run_light_controller import load_functions
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

load_functions(ROOT, 'scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py',
               ['validate_recovery'], globals())

CANONICAL = 'cfm_ts_original__main_text__node_e400__pendulum__seed1104'
ORIGINAL = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
PRIOR = 'configs/experiments/fm_cfm_ts_exact_recovery_queue.v7.json'
PREDECESSOR = 'configs/runtime/fm_cfm_ts_preserved_seed1103_retry.2026-10-12_v1.json'
QUEUE = 'configs/experiments/fm_cfm_seed1104_actual99_full_retry_queue.v1.json'
RUNTIME = 'configs/runtime/fm_cfm_seed1104_actual99_full_retry.v1.json'
STATE = 'experiments/runs/fm_cfm_seed1104_actual99_full_retry_v1'
EVIDENCE = 'projects/flow_matching_research/results/2026-10-13/cfm_seed1104_actual99_full_retry_v1'
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


def validate_retry(original, retry):
    validate_recovery(original, retry)
    if len(original['jobs']) != 285 or retry.get('additional_training_seed_slots') != 0:
        raise ValueError('All original slots and seed accounting must remain')
    changed = [j for j, old in zip(retry['jobs'], original['jobs']) if j != old]
    if len(changed) != 1 or changed[0].get('recovery_original_job_id') != CANONICAL:
        raise ValueError('Only the terminal seed1104 slot may be redirected')
    j = changed[0]
    if (j['epochs'] != 400 or j['expected_optimizer_updates'] != 400
            or j['batch_size'] != 258 or j['seed'] != 1104
            or j['test_points_per_trajectory'] != 500):
        raise ValueError('Full original training and evaluation required')


def main():
    targets = [ROOT / p for p in (QUEUE, RUNTIME, STATE, EVIDENCE)]
    if any(p.exists() for p in targets):
        raise FileExistsError('Preserve existing registrations and attempts')
    original, prior, predecessor = read(ROOT / ORIGINAL), read(ROOT / PRIOR), read(ROOT / PREDECESSOR)
    binding(predecessor, 'recovery')
    old = next(j for j in prior['jobs'] if j.get('recovery_original_job_id') == CANONICAL)
    receipt_path = prior['state_root'] + '/receipts/' + old['id'] + '.json'
    progress_path = old['output_directory'] + '/progress.json'
    receipt, progress = read(ROOT / receipt_path), read(ROOT / progress_path)
    if (receipt['exit_code'] == 0 or receipt['complete']
            or not receipt['resource_observations']['resource_guard_aborted']
            or progress['epoch'] != 99 or progress['required_epochs'] != 400):
        raise ValueError('Actual terminal 99/400 memory abort required')
    previous_owner = read(ROOT / prior['state_root'] / 'status.json')
    if previous_owner['state'] != 'completed_with_unresolved_jobs' or previous_owner['active_pid'] is not None:
        raise ValueError('Original recovery controller must be terminal')
    if psutil.pid_exists(previous_owner['pid']):
        raise ValueError('Previous recovery controller still exists; review identity')
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(CANONICAL in arg for arg in (proc.info['cmdline'] or [])):
                raise ValueError('A matching original seed handle is live')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Check every registered alias, rather than choosing a favorable later attempt.
    for path in (ROOT / 'configs/experiments').glob('fm_cfm*.json'):
        for job in read(path).get('jobs', []):
            if job.get('recovery_original_job_id', job['id']) == CANONICAL:
                if (ROOT / job['output_directory'] / 'result.json').exists():
                    raise ValueError('Existing result needs full completion review before retry')
    original_files = []
    for directory in (ROOT / old['output_directory'], ROOT / old['artifact_directory']):
        if directory.exists():
            original_files += [dict(path=str(p), sha256=sha(p)) for p in directory.rglob('*') if p.is_file()]
    if any(Path(f['path']).suffix in ('.pt', '.pth') for f in original_files):
        raise ValueError('A persisted model requires checkpoint review before fresh retry')
    queue = deepcopy(original)
    job = next(j for j in queue['jobs'] if j['id'] == CANONICAL)
    job.update(id=CANONICAL + '__actual99_full_retry_v1', recovery_original_job_id=CANONICAL,
               output_directory=STATE + '/jobs/' + CANONICAL,
               artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_seed1104_actual99_full_retry_v1/' + CANONICAL)
    evidence = dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_scope=SCOPE,
                    previous_terminal_controller=previous_owner, terminal_receipt=receipt,
                    preserved_progress=progress, original_files=original_files,
                    completed_epochs_before_abort=99, required_original_epochs=400,
                    credited_previous_epochs_for_fresh_retry=0, checkpoint_resume_claimed=False,
                    additional_training_seed_slots=0, benchmark_performance=False,
                    all_original_experiments_complete=False)
    pins = [ORIGINAL, PRIOR, PREDECESSOR, receipt_path, progress_path,
            Path(__file__).relative_to(ROOT).as_posix(),
            'scripts/flow_matching/run_cfm_ts_stricter_admission_retry_v1.py',
            'scripts/flow_matching/run_preserved_native_gan_retry_v1.py',
            'scripts/flow_matching/run_cfm_ts_receipt_runtime_v3.py']
    extra = [dict(path=p, sha256=sha(ROOT / p)) for p in pins]
    queue.update(state_root=STATE, recovery_original_queue=ORIGINAL,
                 recovery_original_queue_sha256=sha(ROOT / ORIGINAL),
                 recovery_job_ids=[job['id']], recovery_evidence=[evidence],
                 additional_training_seed_slots=0,
                 registered_utc=evidence['captured_utc'])
    queue['source_receipts'] += extra
    validate_retry(original, queue)
    exclusive_json(ROOT / QUEUE, queue)
    runtime = deepcopy(predecessor)
    runtime['controllers']['recovery'] = dict(queue=QUEUE, queue_sha256=sha(ROOT / QUEUE), selected_job_ids=[job['id']])
    runtime['source_receipts'] += extra
    runtime['retry_admission_commit_headroom_bytes'] = 40 * 1024**3
    runtime['boundary'] = ('All285original slots preserved. Only terminal99/400 NODE/Pendulum/seed1104 gets a fresh '
                           'full400epoch attempt. Original batch258, grid160, full500test, source, seed, optimizer, '
                           'ODE tolerances, emergency guards and family lock unchanged. Admission40GiB only. '
                           'No partial epoch credit, checkpoint resume, added seed or benchmark performance.')
    binding(runtime, 'recovery')
    exclusive_json(ROOT / RUNTIME, runtime)
    exclusive_json(ROOT / EVIDENCE / 'registration.json', dict(evidence, source_receipts=extra, runtime=RUNTIME, queue=QUEUE))
    exclusive_json(ROOT / 'projects/flow_matching_research/ara/execution/cfm_seed1104_actual99_full_retry_registered.v1.json',
                   dict(original_scope=SCOPE, required_canonical_cfm_jobs=285, runtime=RUNTIME, queue=QUEUE,
                        selected_original_id=CANONICAL, original_required_epochs=400, previous_partial_epochs=99,
                        fresh_retry_previous_epoch_credit=0, all_original_experiments_complete=False,
                        paper_equivalence_certified=False, tab_equivalence_certified=False))
    print(json.dumps(dict(runtime=RUNTIME, queue=QUEUE, selected=job['id'], full_original_slots=285)))


if __name__ == '__main__':
    main()
