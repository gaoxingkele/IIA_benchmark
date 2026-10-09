"""Exact-budget recovery registration for pre-training CUDA allocator init failure."""
import json
from scripts.flow_matching.grasp_protocol_runner import ROOT,sha,verify,write_json


def main():
    config_path='configs/reproducibility/grasp_cuda_allocator_init_repair.v1.json'
    config=json.loads((ROOT/config_path).read_text(encoding='utf-8'))
    path=ROOT/config['queue']
    if path.exists():
        raise FileExistsError('Reviewed recovery queue already frozen')
    source=json.loads((ROOT/config['original_queue']).read_text(encoding='utf-8'))
    if sha(ROOT/config['original_queue'])!=config['original_queue_sha256']:
        raise ValueError('Original frozen queue changed')
    verify(source)
    queue={**source,'state_root':config['state_root'],'artifact_root':config['artifact_root'],
           'reviewed_runtime_repair':config_path,'source_receipts':list(source['source_receipts'])}
    jobs=[]
    for original in source['jobs']:
        ident=original['id'].replace('grasp_protocol_v2__','grasp_protocol_v3__',1)
        job={**original,'id':ident,'original_id':original['id'],
             'output_directory':config['state_root']+'/jobs/'+ident}
        assert {k:v for k,v in job.items() if k not in ['id','original_id','output_directory']}=={
                k:v for k,v in original.items() if k not in ['id','output_directory']}
        jobs.append(job)
    queue['jobs']=jobs
    paths=[config_path,config['original_queue'],'scripts/flow_matching/grasp_protocol_cuda_runtime.py',
           'scripts/flow_matching/run_grasp_protocol_queue_v3.py','scripts/flow_matching/summarize_grasp_protocol_v3.py',
           'scripts/flow_matching/register_grasp_cuda_runtime.py','tests/test_grasp_cuda_runtime.py']
    queue['source_receipts'] += [{'path':p,'sha256':sha(ROOT/p)} for p in paths]
    verify(queue)
    write_json(path,queue)
    print(json.dumps({'jobs':len(jobs),'cohorts':queue['cohort_count'],'queue_sha256':sha(path),
                     'unchanged_training_data_architecture_budgets_seeds':True}))


if __name__=='__main__':
    main()
