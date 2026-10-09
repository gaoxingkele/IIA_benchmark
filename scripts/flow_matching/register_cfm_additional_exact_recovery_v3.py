"""Register newly resource-aborted CFM slots, retaining all original budgets."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, read, sha, write, verify
from scripts.flow_matching.run_light_controller import load_functions

load_functions(ROOT, 'scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py',
               ['validate_recovery'], globals())


def main():
    original_path = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
    target = 'configs/experiments/fm_cfm_ts_exact_recovery_queue.v3.json'
    runtime_target = 'configs/runtime/fm_cfm_ts_low_memory_controller.v5.json'
    if (ROOT / target).exists() or (ROOT / runtime_target).exists():
        raise FileExistsError('Preserve exact-budget retry registration')
    original = read(ROOT / original_path); verify(original)
    proof_path = 'projects/flow_matching_research/results/2026-10-09/cfm_exact_recovery_1215/execution_audit.json'
    proof = read(ROOT / proof_path)
    if proof['queue_sha256'] != sha(ROOT / original_path):
        raise ValueError('Canonical completed-slot proof belongs to another queue')
    resolved = set()
    for row in proof['jobs']:
        if row['status'] == 'completed':
            if sha(ROOT / row['result_path']) != row['result_sha256']:
                raise ValueError('Previously verified full result changed')
            resolved.add(row['id'])
    queue = deepcopy(original); selected = []; evidence = []
    for job in queue['jobs']:
        path = ROOT / original['state_root'] / 'receipts' / (job['id'] + '.json')
        if job['id'] in resolved or not path.exists():
            continue
        receipt = read(path)
        if receipt['complete'] or not receipt['resource_observations'].get('resource_guard_aborted'):
            continue
        if (ROOT / job['output_directory'] / 'result.json').exists():
            raise ValueError('Do not retry a result without fresh full completion validation')
        old = job['id']
        evidence.append({'original_id': old, 'receipt': path.relative_to(ROOT).as_posix(),
            'receipt_sha256': sha(path), 'resource_observations': receipt['resource_observations'],
            'original_output_preserved': job['output_directory']})
        job.update(id=old + '__exact_recovery_v2', recovery_original_job_id=old,
            output_directory='experiments/runs/fm_cfm_ts_exact_recovery_v2/jobs/' + old,
            artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_ts_exact_recovery_v2/' + old)
        selected.append(job['id'])
    if not selected:
        raise ValueError('No new authoritative resource-aborted job')
    queue.update(state_root='experiments/runs/fm_cfm_ts_exact_recovery_v2',
        recovery_original_queue=original_path, recovery_original_queue_sha256=sha(ROOT / original_path),
        recovery_job_ids=selected, recovery_evidence=evidence,
        recovery_selection='First full verified exact-budget recovery resolves original slot; no metric-based selection.',
        prior_resolved_slot_proof={'path': proof_path, 'sha256': sha(ROOT / proof_path)},
        registered_utc=datetime.now(timezone.utc).isoformat())
    extra = [{'path': Path(__file__).relative_to(ROOT).as_posix(), 'sha256': sha(Path(__file__))}]
    queue['source_receipts'] += extra
    validate_recovery(original, queue); write(ROOT / target, queue)
    runtime = deepcopy(read(ROOT / 'configs/runtime/fm_cfm_ts_low_memory_controller.v4.json'))
    for source in runtime['source_receipts']:
        if sha(ROOT / source['path']) != source['sha256']:
            raise ValueError('Live scheduler source changed')
    runtime['controllers']['recovery'] = {'queue': target, 'queue_sha256': sha(ROOT / target),
                                        'selected_job_ids': selected}
    runtime['source_receipts'] += extra
    runtime['boundary'] = 'Additional exact full-budget recovery after new authoritative memory guard exit. Original285 jobs, full data, seeds and scientific budgets unchanged. Shared CFM family lock and original start/emergency thresholds remain enforced.'
    write(ROOT / runtime_target, runtime)
    print(json.dumps({'queue': target, 'runtime': runtime_target, 'selected_full_budget_jobs': selected,
                      'canonical_scope': len(queue['jobs'])}))


if __name__ == '__main__':
    main()
