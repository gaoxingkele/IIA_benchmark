"""Freeze stronger resource policy, retaining all 140 original full GiFlow jobs."""
import argparse
import json
from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json


def register(policy_path, root=ROOT):
    policy = json.loads((root / policy_path).read_text(encoding='utf-8'))
    original_path = root / policy['original_queue']
    if sha(original_path) != policy['original_queue_sha256']:
        raise ValueError('Original GiFlow queue changed')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    verify(original, root)
    target = root / policy['queue']
    if target.exists():
        raise FileExistsError('Native serial queue already frozen')
    queue = {**original, 'state_root': policy['state_root'], 'resource_lock': policy['resource_lock'],
             'settings': {**original['settings'], **policy['settings_overrides']},
             'cohort_count': len(original['jobs']), 'cohort_unit': 'one full dataset graph per seed and recipe',
             'memory_preflight_config': policy['memory_preflight_config'],
             'memory_preflight_config_sha256': sha(root / policy['memory_preflight_config']),
             'reviewed_scheduler_policy': policy_path, 'source_receipts': list(original['source_receipts'])}
    for key, value in original['settings'].items():
        if key.startswith(('minimum_', 'emergency_')) and queue['settings'][key] < value:
            raise ValueError('Original resource floor weakened')
        if key == 'cpu_threads' and queue['settings'][key] != value:
            raise ValueError('Original thread budget changed')
    paths = [policy_path, policy['original_queue'], policy['memory_preflight_config'],
             'scripts/flow_matching/run_grasp_protocol_queue.py',
             'scripts/flow_matching/run_giflow_guarded_gpu_queue_v2.py',
             'scripts/flow_matching/run_giflow_serial_native_queue_v3.py',
             'scripts/flow_matching/register_giflow_serial_native_queue.py',
             'tests/test_fm_giflow_serial_native_queue.py']
    queue['source_receipts'] += [{'path': p, 'sha256': sha(root / p)} for p in paths]
    verify(queue, root)
    write_json(target, queue)
    return queue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    cli = parser.parse_args()
    queue = register(cli.policy)
    print(json.dumps({'original_jobs_unchanged': len(queue['jobs']), 'resource_settings': queue['settings']}))
