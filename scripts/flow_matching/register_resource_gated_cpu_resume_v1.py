"""Register per-job memory admission for four unchanged, full-scope CPU queues."""
from datetime import datetime, timezone
import json

from scripts.flow_matching.run_light_controller import ROOT, digest


def main():
    original_path = 'configs/runtime/fm_cpu_boundary_handoff.v1.json'
    destination = ROOT / 'configs/runtime/fm_resource_gated_cpu_resume.v1.json'
    if destination.exists():
        raise FileExistsError('Keep resource admission registration immutable')
    original = json.loads((ROOT / original_path).read_text(encoding='utf-8'))
    receipts = list(original['source_receipts'])
    for entry in receipts:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen full CPU source differs: ' + entry['path'])
    for path in [original_path, 'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
            'scripts/flow_matching/register_resource_gated_cpu_resume_v1.py',
            'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
            'scripts/flow_matching/prepare_cfm_ts_original_data.py',
            'scripts/flow_matching/run_cfm_ts_original_job.py',
            'tests/test_fm_resource_gated_light_controller_v1.py']:
        if path not in {r['path'] for r in receipts}:
            receipts.append(dict(path=path, sha256=digest(ROOT / path)))
    registration = dict(schema_version=1, original_runtime=original_path,
        source_receipts=receipts, registered_utc=datetime.now(timezone.utc).isoformat(),
        resource_lock='experiments/runs/fm_heavy_recovery_cpu.lock',
        state_root='experiments/runs/fm_resource_gated_cpu_resume_v1',
        controllers={b['name']: dict(queue=b['queue_path'], queue_sha256=b['queue_sha256'],
            registered_jobs=b['registered_jobs']) for b in original['controllers']},
        settings=dict(minimum_available_memory_bytes=12*1024**3,
            minimum_commit_headroom_bytes=32*1024**3, emergency_commit_headroom_bytes=16*1024**3,
            guard_poll_seconds=2, controller_yield_seconds=2),
        boundary='Per-job shared CPU admission and emergency guard only. Original queues, model commands, '
            'datasets, trained seeds, hyperparameters, full budgets and completed-result verifiers are unchanged. '
            'Partial attempts remain intact and require separately registered exact retries. '
            'Observation timeout does not terminate a live job.')
    destination.write_text(json.dumps(registration, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(config=destination.relative_to(ROOT).as_posix(),
        registered_jobs=sum(b['registered_jobs'] for b in original['controllers']),
        controller_count=len(original['controllers']))))


if __name__ == '__main__':
    main()
