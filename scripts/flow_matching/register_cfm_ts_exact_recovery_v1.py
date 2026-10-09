"""Register full-budget retries only for authoritative resource-aborted jobs."""
from copy import deepcopy
import json
import psutil

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT,read,sha
from scripts.flow_matching.run_cfm_ts_original_job import verify,write


def validate_recovery(original,recovery):
    verify(original);verify(recovery)
    if original['resource_policy']!=recovery['resource_policy']:
        raise ValueError('Original memory safeguards changed')
    source={j['id']:j for j in original['jobs']}
    if len(recovery['jobs'])!=len(source):raise ValueError('Full original slot scope differs')
    permitted={'id','output_directory','artifact_directory','recovery_original_job_id'}
    originals=[]
    for candidate in recovery['jobs']:
        identity=candidate.get('recovery_original_job_id',candidate['id']);old=source[identity]
        if any(old.get(k)!=candidate.get(k) for k in set(old)|set(candidate) if k not in permitted):
            raise ValueError('Exact retry changes original scientific protocol')
        if candidate['id']!=identity and (candidate['output_directory']==old['output_directory'] or candidate['artifact_directory']==old['artifact_directory']):
            raise ValueError('Retry would overwrite original artifacts')
        originals.append(identity)
    if set(originals)!=set(source) or len(set(originals))!=len(originals):raise ValueError('Canonical slots duplicated/missing')


def main():
    original_path='configs/experiments/fm_cfm_ts_original_queue.v1.json'
    recovery_path='configs/experiments/fm_cfm_ts_exact_recovery_queue.v1.json'
    runtime_path='configs/runtime/fm_cfm_ts_low_memory_controller.v2.json'
    if (ROOT/recovery_path).exists() or (ROOT/runtime_path).exists():raise FileExistsError('Preserve frozen recovery registration')
    original=read(ROOT/original_path);verify(original);queue=deepcopy(original)
    selected=[];evidence=[]
    for job in queue['jobs']:
        receipt=ROOT/original['state_root']/'receipts'/f"{job['id']}.json"
        if not receipt.exists():continue
        status=read(receipt)
        if status['complete'] or not status['resource_observations'].get('resource_guard_aborted'):continue
        old=job['id'];job.update(id=old+'__exact_recovery_v1',recovery_original_job_id=old,
            output_directory='experiments/runs/fm_cfm_ts_exact_recovery_v1/jobs/'+old,
            artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_ts_exact_recovery_v1/'+old)
        selected.append(job['id']);evidence.append({'original_id':old,'receipt':receipt.relative_to(ROOT).as_posix(),
            'receipt_sha256':sha(receipt),'original_output_preserved':next(j['output_directory'] for j in original['jobs'] if j['id']==old),
            'resource_observations':status['resource_observations']})
    if not selected:raise ValueError('No authoritative resource-aborted jobs to recover')
    queue.update(state_root='experiments/runs/fm_cfm_ts_exact_recovery_v1',recovery_original_queue=original_path,
        recovery_original_queue_sha256=sha(ROOT/original_path),recovery_job_ids=selected,
        recovery_evidence=evidence,recovery_selection='First full verified exact-budget attempt resolves the original slot; no metric-dependent selection')
    files=['scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py',
        'scripts/flow_matching/transition_cfm_ts_low_memory_v2.py','tests/test_fm_cfm_ts_low_memory_v2.py']
    receipts=[{'path':p,'sha256':sha(ROOT/p)} for p in files]
    queue['source_receipts']+=receipts
    validate_recovery(original,queue);write(ROOT/recovery_path,queue)
    state=read(ROOT/original['state_root']/'status.json');process=psutil.Process(state['pid'])
    if 'scripts.flow_matching.run_cfm_ts_light_queue_v1' not in process.cmdline():raise ValueError('Predecessor identity differs')
    runtime={'controllers':{'original':{'queue':original_path,'queue_sha256':sha(ROOT/original_path)},
        'recovery':{'queue':recovery_path,'queue_sha256':sha(ROOT/recovery_path),'selected_job_ids':selected}},
        'family_lock':'experiments/runs/fm_cfm_ts_original_cpu.lock',
        'source_receipts':receipts+original['source_receipts'],
        'predecessor':{'pid':process.pid,'create_time':process.create_time(),'command':process.cmdline(),
            'status_path':original['state_root']+'/status.json'},
        'handoff_root':'experiments/runs/fm_cfm_ts_low_memory_transition_v2',
        'boundary':'Original full285 jobs, data, model, worker, seeds, epochs, optimizer updates and resource thresholds unchanged. Isolated verification avoids NumPy in the waiting parent. Two exact-budget retries remain separate preserved attempts.'}
    write(ROOT/runtime_path,runtime)
    print(json.dumps({'full_original_slots':len(queue['jobs']),'resource_aborted_retry_jobs':len(selected),'runtime':runtime_path}))


if __name__=='__main__':main()
