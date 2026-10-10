"""Register untouched full GRASP jobs and all seven outstanding full-result audits."""
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.grasp_protocol_runner import complete,verify


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(relative,value):
    path=ROOT/relative
    if path.exists():raise FileExistsError('Preserve frozen registration: '+relative)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def receipts(paths):
    return [dict(path=p,sha256=digest(ROOT/p)) for p in dict.fromkeys(paths)]


def main():
    queue_path='configs/experiments/fm_grasp_protocol_queue.v4.json'
    queue=read(ROOT/queue_path);verify(queue)
    source='configs/reproducibility/fm_grasp_seed1104_full_audit.2026-10-11_v1.json'
    previous=read(ROOT/source)
    profiles=['tau_4','tau_8']
    jobs=[next(j for j in queue['jobs'] if j['profile']==profile and j['dataset']=='swan' and j['seed']==1103) for profile in profiles]
    if not all(complete(j,ROOT) for j in jobs):raise ValueError('Only complete original full-budget jobs may be audited')
    audit=dict(previous,full_results=[dict(id=j['id'],result_path=j['output_directory']+'/result.json',
        result_sha256=digest(ROOT/j['output_directory']/'result.json')) for j in jobs],
        output_directory='projects/flow_matching_research/results/2026-10-11/grasp_tau4_tau8_full_audit_v2',
        state_root='experiments/runs/fm_grasp_tau4_tau8_full_audit_v2',
        boundary='Two completed1500-epoch/93000-update full SWAN tau4/tau8 seed1103 ablations. All original arrays, rows, labels, checkpoint bindings and score metrics required. No extra seed or author-equivalence certification.')
    paths=[r['path'] for r in previous['source_receipts'] if '/jobs/' not in r['path']]
    paths.extend(r['result_path'] for r in audit['full_results'])
    audit['source_receipts']=receipts(paths)
    new_audit='configs/reproducibility/fm_grasp_tau4_tau8_full_audit.2026-10-11_v2.json'
    write_new(new_audit,audit)
    names=['tau_0_0p5_1_seed1103','gnn_main_seed1104','tau_4_8_seed1103']
    configs=['configs/reproducibility/fm_grasp_new_full_ablations_audit.2026-10-10_v1.json',source,new_audit]
    bindings=[dict(name=name,config=path,config_sha256=digest(ROOT/path),
                   output_directory=read(ROOT/path)['output_directory']) for name,path in zip(names,configs)]
    sources=['scripts/flow_matching/run_bounded_readonly_grasp_audits_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/register_grasp_gpu_and_readonly_resume_v2.py',
        'tests/test_fm_grasp_bounded_readonly_audits_v2.py',
        'tests/test_fm_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_tsad_queue.py',*configs]
    sources.extend(r['path'] for p in configs for r in read(ROOT/p)['source_receipts'])
    bounded=dict(state_root='experiments/runs/fm_grasp_bounded_readonly_audits_v2',
        resource_lock='experiments/runs/fm_readonly_checkpoint_audit_cpu.lock',audits=bindings,full_result_count=7,
        settings=dict(minimum_available_memory_bytes=12*1024**3,minimum_commit_headroom_bytes=22*1024**3,
            emergency_commit_headroom_bytes=16*1024**3,maximum_worker_rss_bytes=2*1024**3,
            maximum_worker_private_bytes=4*1024**3),source_receipts=receipts(sources),
        boundary='Read-only original full-result audit workers; separate audit-family mutex, CUDA disabled and strict per-worker/process-tree memory caps. No training or score-recipe change.')
    audit_runtime='configs/runtime/fm_grasp_bounded_readonly_audits.v2.json'
    write_new(audit_runtime,bounded)
    fair=read(ROOT/'configs/runtime/fm_grasp_full_fairness.v5.json')
    sources=[r['path'] for r in fair['source_receipts']]
    sources.extend(['configs/runtime/fm_grasp_full_fairness.v5.json',
        'scripts/flow_matching/run_grasp_gpu_memory_fair_queue_v6.py',
        'tests/test_fm_grasp_gpu_memory_fair_queue_v6.py',
        'scripts/flow_matching/register_grasp_gpu_and_readonly_resume_v2.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'tests/test_fm_full_gpu_memory_slot_v2.py'])
    gpu=dict(fair,source_receipts=receipts(sources),
        boundary='All9120 original entity jobs/480 original cohorts and all scientific settings unchanged. Same full RAM/commit/VRAM gates and emergency guards, same GPU mutex, original35second post-job fairness yield. CPU training mutex is not GPU memory admission; never switch an active model.')
    gpu_runtime='configs/runtime/fm_grasp_full_gpu_memory_fairness.v6.json'
    write_new(gpu_runtime,gpu)
    print(json.dumps(dict(audit_runtime=audit_runtime,full_results=7,gpu_runtime=gpu_runtime,
                         full_jobs=len(queue['jobs']),cohorts=queue['cohort_count'])))


if __name__=='__main__':main()
