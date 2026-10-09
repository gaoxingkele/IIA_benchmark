"""Freeze scheduling-only successor; retain every original GiFlow experiment slot."""
import json
from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json


def register(policy_path='configs/runtime/fm_giflow_guarded_gpu_lane.v2.json', root=ROOT):
    policy = json.loads((root / policy_path).read_text(encoding='utf-8'))
    source_path = root / policy['original_queue']
    if sha(source_path) != policy['original_queue_sha256']:
        raise ValueError('Frozen original GiFlow queue changed')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    verify(source, root)
    target = root / policy['queue']
    if target.exists():
        raise FileExistsError('GiFlow scheduling queue already frozen')
    if 'gpu_index' in source['settings']:
        raise ValueError('Original GPU settings require a new policy review')
    queue = {**source, 'state_root': policy['state_root'], 'resource_lock': policy['resource_lock'],
             'settings': {**source['settings'], **policy['gpu_settings']},
             'cohort_count': len(source['jobs']), 'cohort_unit': 'one full dataset graph per seed and recipe',
             'reviewed_scheduler_policy': policy_path, 'source_receipts': list(source['source_receipts'])}
    paths = [policy_path, policy['original_queue'], 'scripts/flow_matching/run_grasp_protocol_queue.py',
             'scripts/flow_matching/run_giflow_guarded_gpu_queue_v2.py',
             'scripts/flow_matching/register_giflow_guarded_gpu_lane.py', 'tests/test_fm_giflow_gpu_lane.py']
    queue['source_receipts'] += [{'path': p, 'sha256': sha(root / p)} for p in paths]
    for key in ['jobs', 'python', 'environment_lock', 'prediction_artifact_root', 'data_config', 'runtime_config']:
        assert queue[key] == source[key]
    assert all(queue['settings'][k] == v for k, v in source['settings'].items())
    verify(queue, root)
    write_json(target, queue)
    print(json.dumps({'registered_jobs':len(queue['jobs']), 'queue_sha256':sha(target),
                     'all_original_slots_and_budgets_unchanged':True, 'shared_grasp_gpu_mutex':queue['resource_lock']}))
    return queue


if __name__ == '__main__':
    register()
