"""Register immutable native retries for explicitly recorded terminal failures."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import psutil

from scripts.flow_matching.native_exact_recovery import ROOT, components, sha, verify
from scripts.flow_matching.grasp_protocol_runner import write_json


def register(policy_path, root=ROOT):
    policy = json.loads((root / policy_path).read_text(encoding='utf-8'))
    original_path = root / policy['original_queue']
    if sha(original_path) != policy['original_queue_sha256']:
        raise ValueError('Original native queue changed')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    worker, _ = components(policy['track'])
    worker.verify(original, root)
    new_path = root / policy['queue']
    if new_path.exists():
        raise FileExistsError('Recovery registration is already frozen')
    queue = deepcopy(original)
    queue.update(recovery_track=policy['track'], recovery_original_queue=policy['original_queue'],
                 recovery_original_queue_sha256=policy['original_queue_sha256'],
                 recovery_original_job_ids=policy['original_job_ids'], jobs=[],
                 recovery_failure_receipts=[], state_root=policy['state_root'],
                 artifact_root=policy['artifact_root'], lock_poll_seconds=.25,
                 cohort_count=len(policy['original_job_ids']), recovery_policy=policy['policy'])
    active = []
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'].lower() != 'python.exe':
            continue
        try:
            active.append(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for old_id in policy['original_job_ids']:
        original_job = next(j for j in original['jobs'] if j['id'] == old_id)
        failure = root / original_job['output_directory'] / 'failure.json'
        if worker.complete(original_job, root) or not failure.exists():
            raise ValueError('Only terminal failed original slots can be recovered: ' + old_id)
        if any(old_id in command for command in active):
            raise ValueError('Original model remains live; cannot duplicate its experiment')
        job = deepcopy(original_job)
        job.update(id=old_id + '__native_exact_recovery_v1', recovery_original_job_id=old_id,
                   output_directory=policy['state_root'] + '/jobs/' + old_id + '__native_exact_recovery_v1')
        if (root / job['output_directory']).exists():
            raise FileExistsError('Recovery output already exists')
        queue['jobs'].append(job)
        queue['recovery_failure_receipts'].append({'original_job_id': old_id,
            'path': failure.relative_to(root).as_posix(), 'sha256': sha(failure)})
    paths = [policy_path, policy['original_queue'], 'scripts/flow_matching/native_exact_recovery.py',
             'scripts/flow_matching/run_native_exact_recovery_job.py',
             'scripts/flow_matching/run_native_exact_recovery_queue.py',
             'scripts/flow_matching/register_native_exact_recovery.py', 'tests/test_fm_native_exact_recovery.py']
    queue['source_receipts'] += [{'path': p, 'sha256': sha(root / p)} for p in paths]
    verify(queue, root)
    write_json(new_path, queue)
    return queue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    cli = parser.parse_args()
    queue = register(cli.policy)
    print(json.dumps({'registered_recoveries': len(queue['jobs']), 'original_slots': queue['recovery_original_job_ids']}))
