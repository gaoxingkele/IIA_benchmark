"""Admit all413 full original SHCL commands with CPU/commit/GPU safeguards."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching.run_shcl_native_job_v1 import observer,verify


def gpu_snapshot(settings):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
        encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip().splitlines()
    rows=[line.split(',') for line in lines]
    match=[row for row in rows if row[0].strip()==settings['gpu_uuid']]
    if len(match)!=1: raise ValueError('Exact registered CUDA UUID required')
    return dict(gpu_uuid=settings['gpu_uuid'],gpu_free_bytes=int(match[0][1].strip())*2**20)


def complete(job):
    path=Path(job['output_directory'])/'observer/result.json'
    if not path.exists(): return False
    result=observer.native.read(path)
    if result['job']!=job or not result['passed'] or result.get('synthetic_operator_only_not_benchmark_performance'):
        raise ValueError('Wrong native source job or diagnostic result')
    if job['mode']=='train':
        if (not result['original_training_complete'] or result['optimizer_updates']!=job['expected_optimizer_updates']
                or len(result['epochs'])!=job['num_epochs']): raise ValueError('Full native training budget required')
        for epoch in result['epochs']:
            if observer.native.sha(epoch['full_state'])!=epoch['full_state_sha256']: raise ValueError('Full original optimizer/RNG checkpoint changed')
    else:
        if not result['full_native_test_complete']: raise ValueError('Full native test windows required')
        for row in result['full_source_score_frames']:
            if observer.native.sha(row['context'])!=row['context_sha256'] or observer.native.sha(row['arrays'])!=row['arrays_sha256']:
                raise ValueError('Complete native context/loss/score arrays changed')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--queue',required=True)
    args=parser.parse_args(); queue=observer.native.read(observer.native.ROOT/args.queue); verify(queue)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api(); api['resource_snapshot']=resource_snapshot
    base=observer.native.ROOT/queue['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Same complete native413 queue already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),
        queue_sha256=observer.native.sha(observer.native.ROOT/args.queue),jobs={},active_pid=None,full_original_commands=413,
        all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat()); observer.write(base/'status.json',state)
    @contextmanager
    def reservation():
        while True:
            snapshot=dict(resource_snapshot(),**gpu_snapshot(queue['settings']))
            ready=api['has_headroom'](snapshot,queue['settings']) and snapshot['gpu_free_bytes']>=queue['settings']['minimum_gpu_free_bytes']
            handle=api['exclusive_lock'](observer.native.ROOT/queue['resource_lock']) if ready else None
            if handle is not None:
                confirmed=dict(resource_snapshot(),**gpu_snapshot(queue['settings']))
                if api['has_headroom'](confirmed,queue['settings']) and confirmed['gpu_free_bytes']>=queue['settings']['minimum_gpu_free_bytes']:
                    try: yield confirmed
                    finally: handle.close()
                    return
                handle.close()
            update(dict(snapshot,state='waiting_for_memory_or_full_native_GPU_capacity' if not ready else 'waiting_for_native_lane',shared_slot_owned=False))
            time.sleep(10)
    try:
        update({})
        for job in queue['jobs']:
            if complete(job): state['jobs'][job['id']]='source_command_completed_checkpoint_replay_pending'; continue
            if Path(job['output_directory']).exists(): state['jobs'][job['id']]='failed_or_partial_preserved'; continue
            if job['depends_on'] and not complete(next(j for j in queue['jobs'] if j['id']==job['depends_on'])):
                state['jobs'][job['id']]='awaiting_complete_original_training'; continue
            update(dict(state='awaiting_full_native_capacity',active_job=job['id']))
            with reservation():
                command=[queue['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_shcl_native_job_v1',
                    '--queue',args.queue,'--job-id',job['id']]
                logs=base/'logs'; logs.mkdir(exist_ok=True)
                with (logs/(job['id']+'.stdout.log')).open('xb') as out,(logs/(job['id']+'.stderr.log')).open('xb') as err:
                    child=subprocess.Popen(command,cwd=observer.native.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=queue['settings']['gpu_uuid'],
                            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    update(dict(state='running_full_native_source_command',active_pid=child.pid,
                        active_create_time=psutil.Process(child.pid).create_time(),worker_command=command))
                    gpu_abort={}
                    def guarded_update(values):
                        current=gpu_snapshot(queue['settings'])
                        if current['gpu_free_bytes']<queue['settings']['emergency_gpu_free_bytes']:
                            gpu_abort.update(resource_guard_aborted=True,abort_gpu_snapshot=current); api['terminate_tree'](child)
                        update(dict(values,**current))
                    code,usage=capped_wait(child,queue,api,guarded_update)
                passed=code==0 and not usage['resource_guard_aborted'] and not gpu_abort and complete(job)
                observer.write(base/'receipts'/(job['id']+'.json'),dict(usage,**gpu_abort,exit_code=code,passed=passed))
                state['jobs'][job['id']]='source_command_completed_checkpoint_replay_pending' if passed else 'failed_source_command_preserved'
                update(dict(active_pid=None,active_job=None))
        update(dict(state='full_source_queue_iteration_finished_all_fidelity_and_TAB_gates_still_required'))
    finally: owner.close()


if __name__=='__main__':main()
