"""Freeze every completed canonical slot for full trained-model re-inference."""
import json
from pathlib import Path
import sys

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha


def main():
    capture = 'projects/flow_matching_research/results/2026-10-11/cfm_checkpoint_replay_v1'
    snapshot = read(ROOT/capture/'execution_audit.json')
    original_path = 'configs/experiments/fm_cfm_ts_original_queue.v1.json'
    original = read(ROOT/original_path)
    queue_paths = [original_path,
        'configs/experiments/fm_cfm_ts_exact_recovery_queue.v2.json',
        'configs/experiments/fm_cfm_ts_exact_recovery_queue.v3.json',
        'configs/experiments/fm_cfm_ts_exact_recovery_queue.v4.json']
    all_jobs = {}
    sources = {capture+'/execution_audit.json', 'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/register_cfm_checkpoint_replay_v1.py', 'tests/test_fm_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py', 'scripts/flow_matching/model_registry.py',
        'scripts/flow_matching/run_cfm_ts_original_job.py', 'scripts/flow_matching/register_cfm_ts_original.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py', 'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/heavy_resources.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_tsad_queue.py', original['environment_lock']}
    for path in queue_paths:
        queue = read(ROOT/path)
        sources.add(path)
        sources.update(r['path'] for r in queue['source_receipts'])
        for job in queue['jobs']:
            all_jobs[job['id']] = job
    records = []
    for row in snapshot['jobs']:
        if row['status'] != 'completed':
            continue
        job = all_jobs[row['execution_id']]
        result = read(ROOT/row['result_path'])
        if result['job'] != job or sha(ROOT/row['result_path']) != row['result_sha256']:
            raise ValueError('Completed checkpoint belongs to another immutable job')
        sources.add(row['result_path'])
        records.append(dict(original_canonical_id=row['id'], result_path=row['result_path'],
            result_sha256=row['result_sha256'], job=job))
    if len(records) != snapshot['job_counts']['completed'] or len({r['original_canonical_id'] for r in records}) != len(records):
        raise ValueError('All completed canonical slots, without duplicate recovery seeds, required')
    config = dict(full_job_count=len(records), full_results=records,
        environment_lock=original['environment_lock'], python=sys.executable, cpu_threads=2,
        prediction_tolerance=dict(rtol=1e-7, atol=1e-8),
        state_root='experiments/runs/fm_cfm_checkpoint_replay_v1',
        output_directory=capture+'/full_checkpoint_replay',
        resource_lock='experiments/runs/fm_readonly_checkpoint_audit_cpu.lock',
        settings=dict(minimum_available_memory_bytes=12*2**30, minimum_commit_headroom_bytes=22*2**30,
            emergency_commit_headroom_bytes=16*2**30, maximum_worker_rss_bytes=2*2**30,
            maximum_worker_private_bytes=4*2**30),
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(sources)],
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        original_canonical_queue=original_path, original_canonical_slots=285,
        boundary='Read-only full-checkpoint inference, separately serialized and bounded to2GiB RSS/4GiB private memory; no CUDA context, model training or optimizer update. Existing CPU training and full GPU resource gates remain unchanged. All frozen completed slots replayed on every original test point with original solver/environment; do not tune tolerance or select seeds by observed agreement.')
    destination = ROOT/'configs/reproducibility/fm_cfm_checkpoint_replay.2026-10-11_v1.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(full_checkpoint_replays=len(records), source_bindings=len(sources), additional_seed_slots=0)))


if __name__ == '__main__':
    main()
