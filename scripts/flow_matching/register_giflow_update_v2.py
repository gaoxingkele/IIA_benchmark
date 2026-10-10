"""Register observation-only GiFlow repair without altering any full experiment."""
import json
from pathlib import Path

import psutil

from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json
from scripts.flow_matching.transition_giflow_at_boundary_v1 import compare_full_jobs


def main():
    original = 'configs/experiments/fm_giflow_native_queue.v3.json'
    successor = 'configs/experiments/fm_giflow_native_queue.v4.json'
    if (ROOT / successor).exists():
        raise FileExistsError('Preserve frozen update-counter queue')
    old = json.loads((ROOT / original).read_text(encoding='utf-8')); verify(old)
    queue = json.loads(json.dumps(old))
    queue.update(id='giflow_native_v4', state_root='experiments/runs/fm_giflow_native_v4',
        prediction_artifact_root='F:/aicoding/IIA_Data/experiments/giflow_native_v4',
        original_canonical_queue=original, original_canonical_queue_sha256=sha(ROOT / original),
        controller_yield_seconds=35,
        observation_repair='Persist actual native train loader, drop_last and per-epoch optimizer updates. Original algorithm, seeds, early stopping and full budgets retained.',
        failure_boundary='Earlier full training/predictions preserved. Lost ephemeral counters are not manufactured or certified; these are new exact full-budget attempts in the same canonical seed slots.')
    models = {}
    for old_path in sorted({j['model_config'] for j in old['jobs']}):
        path = old_path.replace('_v1.json', '_v2.json')
        model = json.loads((ROOT / old_path).read_text(encoding='utf-8'))
        model.update(id=model['id'].replace('_v1', '_v2'),
            implementation='scripts.flow_matching.run_giflow_native_job_v2', queue_config=successor,
            tests=['tests/test_fm_giflow_native.py','tests/test_fm_giflow_update_budget_v2.py','tests/test_fm_giflow_boundary_v1.py'],
            reproduction_status='observed_drop_last_repair_registered_full_native_training_pending')
        if (ROOT / path).exists(): raise FileExistsError('Preserve model config')
        write_json(ROOT / path, model); models[old_path] = path
    for job, original_job in zip(queue['jobs'], old['jobs']):
        job.update(model_config=models[job['model_config']],
            output_directory=queue['state_root']+'/jobs/'+job['id'],
            original_canonical_id=original_job['id'], original_output_directory=original_job['output_directory'])
    compare_full_jobs(old, queue)
    paths = set(models.values()) | {original,
        'scripts/flow_matching/run_giflow_native_job_v2.py','scripts/flow_matching/giflow_update_budget_v2.py',
        'scripts/flow_matching/run_giflow_serial_native_queue_v4.py','scripts/flow_matching/transition_giflow_at_boundary_v1.py',
        'scripts/flow_matching/register_giflow_update_v2.py',
        'tests/test_fm_giflow_update_budget_v2.py','tests/test_fm_giflow_boundary_v1.py'}
    queue['source_receipts'] += [dict(path=p,sha256=sha(ROOT / p)) for p in sorted(paths)]
    verify(queue); write_json(ROOT / successor,queue)
    parent = psutil.Process(21984)
    expected = dict(pid=parent.pid, create_time=parent.create_time(), command=parent.cmdline())
    if expected['create_time'] != 1791643109.8706102 or 'scripts.flow_matching.run_giflow_serial_native_queue_v3' not in expected['command']:
        raise ValueError('Exact currently observed predecessor required')
    handoff = dict(original_queue=original, original_queue_sha256=sha(ROOT / original),
        successor_queue=successor, successor_queue_sha256=sha(ROOT / successor),
        expected_controller=expected, state_root='experiments/runs/fm_giflow_update_boundary_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT / p)) for p in sorted(paths)])
    destination=ROOT/'configs/runtime/fm_giflow_update_boundary.v1.json'
    if destination.exists(): raise FileExistsError('Preserve observed boundary registration')
    write_json(destination,handoff)
    print(json.dumps(dict(full_jobs=len(queue['jobs']), canonical_seed_slots_unchanged=True,
        queue_sha256=sha(ROOT / successor), expected_controller=expected)))


if __name__ == '__main__':
    main()
