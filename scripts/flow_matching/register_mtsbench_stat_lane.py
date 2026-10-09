"""Register a scheduling successor without changing any native experiment slot."""
import json
from scripts.flow_matching.mtsbench_stat_protocol import ROOT,sha,verify,write_json


def register(policy_path,root=ROOT):
    policy=json.loads((root/policy_path).read_text(encoding='utf-8'))
    source_path=root/policy['original_queue']
    if sha(source_path)!=policy['original_queue_sha256']:
        raise ValueError('Frozen original statistical queue changed')
    source=json.loads(source_path.read_text(encoding='utf-8'))
    verify(source,root)
    target=root/policy['queue']
    if target.exists():
        raise FileExistsError('Reviewed statistical queue already frozen')
    queue={**source,'state_root':policy['state_root'],'resource_lock':policy['resource_lock'],
           'reviewed_scheduler_policy':policy_path,'source_receipts':list(source['source_receipts'])}
    paths=[policy_path,policy['original_queue'],'scripts/flow_matching/run_mtsbench_stat_queue_v2.py',
           'scripts/flow_matching/register_mtsbench_stat_lane.py','tests/test_fm_mtsbench_stat_lane.py']
    queue['source_receipts'] += [{'path':p,'sha256':sha(root/p)} for p in paths]
    assert queue['jobs']==source['jobs'] and queue['settings']==source['settings']
    assert queue['artifact_root']==source['artifact_root'] and queue['environment_lock']==source['environment_lock']
    verify(queue,root)
    write_json(target,queue)
    print(json.dumps({'registered_jobs':len(queue['jobs']),'queue_sha256':sha(target),
                     'source_receipts':len(queue['source_receipts']),'all_original_experiment_slots_unchanged':True}))
    return queue


if __name__=='__main__':
    register('configs/runtime/fm_mtsbench_stat_lane.v2.json')
