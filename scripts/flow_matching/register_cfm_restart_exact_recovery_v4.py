"""Retry unresolved reboot/resource-interrupted CFM slots at their exact budgets."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, read, sha, write, verify, external_complete
from scripts.flow_matching.run_light_controller import load_functions

load_functions(ROOT, 'scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py',
               ['validate_recovery'], globals())


def main():
    original_path = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
    target = 'configs/experiments/fm_cfm_ts_exact_recovery_queue.v4.json'
    runtime_target = 'configs/runtime/fm_cfm_ts_low_memory_controller.v6.json'
    if (ROOT / target).exists() or (ROOT / runtime_target).exists():
        raise FileExistsError('Preserve previous exact recovery registrations')
    original = read(ROOT / original_path); verify(original)
    resolved, prior_receipts = set(), []
    for prior_path in ['configs/experiments/fm_cfm_ts_exact_recovery_queue.v2.json',
                       'configs/experiments/fm_cfm_ts_exact_recovery_queue.v3.json']:
        prior = read(ROOT / prior_path); verify(prior)
        for job in prior['jobs']:
            if job['id'] not in prior['recovery_job_ids']:
                continue
            if external_complete(prior_path, prior, job):
                result = ROOT / job['output_directory'] / 'result.json'
                resolved.add(job['recovery_original_job_id'])
                prior_receipts.append(dict(original_id=job['recovery_original_job_id'],
                    queue=prior_path, queue_sha256=sha(ROOT / prior_path),
                    result_path=result.relative_to(ROOT).as_posix(), result_sha256=sha(result),
                    full_artifacts_verified=True))
    boot = psutil.boot_time()
    processes = []
    for process in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        try:
            if process.info['cmdline'] and any('run_cfm_ts_original_job' in s for s in process.info['cmdline']):
                processes.append(process.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    queue = deepcopy(original); selected, evidence = [], []
    for job in queue['jobs']:
        output = ROOT / job['output_directory']
        result = output / 'result.json'
        if job['id'] in resolved or result.exists() or not output.exists():
            continue
        if any(job['id'] in p['cmdline'] for p in processes):
            continue
        receipt_path = ROOT / original['state_root'] / 'receipts' / (job['id'] + '.json')
        receipt = read(receipt_path) if receipt_path.exists() else None
        aborted = bool(receipt and not receipt['complete'] and receipt['resource_observations'].get('resource_guard_aborted'))
        logs = [ROOT / original['state_root'] / 'logs' / (job['id'] + suffix)
                for suffix in ['.stdout.log', '.stderr.log']]
        retained = [p for p in output.rglob('*') if p.is_file()] + [p for p in logs if p.exists()]
        preboot = (receipt is None and output.stat().st_mtime < boot
                   and all(p.stat().st_mtime < boot for p in retained))
        if not aborted and not preboot:
            continue
        identity = job['id']
        evidence.append(dict(original_id=identity,
            reason='authoritative_resource_guard_exit' if aborted else 'preserved_preboot_partial_no_live_worker',
            boot_utc=datetime.fromtimestamp(boot, timezone.utc).isoformat(),
            original_output_preserved=job['output_directory'],
            output_directory_mtime=output.stat().st_mtime,
            receipt_path=receipt_path.relative_to(ROOT).as_posix() if receipt else None,
            receipt_sha256=sha(receipt_path) if receipt else None,
            preserved_files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p),
                mtime=p.stat().st_mtime) for p in retained],
            observed_cfm_model_processes=processes))
        job.update(id=identity + '__exact_recovery_v3', recovery_original_job_id=identity,
            output_directory='experiments/runs/fm_cfm_ts_exact_recovery_v3/jobs/' + identity,
            artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_ts_exact_recovery_v3/' + identity)
        selected.append(job['id'])
    if not selected:
        raise ValueError('No unresolved authoritative interrupted CFM slot')
    queue.update(state_root='experiments/runs/fm_cfm_ts_exact_recovery_v3',
        recovery_original_queue=original_path, recovery_original_queue_sha256=sha(ROOT / original_path),
        recovery_job_ids=selected, recovery_evidence=evidence,
        prior_full_resolutions=prior_receipts, registered_utc=datetime.now(timezone.utc).isoformat(),
        recovery_selection='First full verified exact-budget attempt resolves original slot; no added seed or metric-based selection.')
    extra = dict(path=Path(__file__).relative_to(ROOT).as_posix(), sha256=sha(Path(__file__)))
    queue['source_receipts'].append(extra)
    validate_recovery(original, queue); write(ROOT / target, queue)
    runtime = deepcopy(read(ROOT / 'configs/runtime/fm_cfm_ts_low_memory_controller.v5.json'))
    for source in runtime['source_receipts']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen full CFM controller source differs')
    runtime['controllers']['recovery'] = dict(queue=target, queue_sha256=sha(ROOT / target), selected_job_ids=selected)
    runtime['source_receipts'].append(extra)
    runtime['boundary'] = ('Original285 canonical slots and exact scientific budgets retained. '
        'Only unresolved preboot partials or authoritative resource exits are selected after live process checks '
        'and full validation of earlier recoveries. Original CFM family resource lock and guards remain.')
    write(ROOT / runtime_target, runtime)
    print(json.dumps(dict(queue=target, runtime=runtime_target, selected=selected,
        prior_fully_verified_resolutions=len(resolved), canonical_slots=len(queue['jobs']))))


if __name__ == '__main__':
    main()
