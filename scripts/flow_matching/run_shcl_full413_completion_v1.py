"""Wait for and independently replay every full native SHCL source command."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import time

from scripts.flow_matching.run_shcl_native_replay_job_v1 import replay,verify,complete
from scripts.flow_matching.run_shcl_native_script_queue_v1 import gpu_snapshot
from scripts.flow_matching.timegan_native_common_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); config=replay.native.read(replay.native.ROOT/args.config); queue,model=verify(config)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api(); api['resource_snapshot']=resource_snapshot
    base=replay.native.ROOT/config['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Same full413 completion already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),
        configuration_sha256=replay.native.sha(replay.native.ROOT/args.config),verified={},failures={},
        active_pid=None,full_canonical_commands=413,full_original_training_commands=49,full_original_test_commands=364,
        all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json',state)
    def ready_slot(job):
        while True:
            snapshot=resource_snapshot()
            gpu=gpu_snapshot(config['gpu_settings']) if job['mode']=='test' else {}
            ready=api['has_headroom'](snapshot,config['settings']) and (not gpu or gpu['gpu_free_bytes']>=config['gpu_settings']['minimum_gpu_free_bytes'])
            handle=api['exclusive_lock'](replay.native.ROOT/config['resource_lock']) if ready else None
            if handle is not None:
                check=resource_snapshot(); gpu=gpu_snapshot(config['gpu_settings']) if job['mode']=='test' else {}
                if api['has_headroom'](check,config['settings']) and (not gpu or gpu['gpu_free_bytes']>=config['gpu_settings']['minimum_gpu_free_bytes']): return handle
                handle.close()
            update(dict(snapshot,**gpu,state='waiting_for_original_readonly_capacity',shared_slot_owned=False)); time.sleep(10)
    try:
        update({})
        while True:
            for job in queue['jobs']:
                identifier=job['id']; output=Path(config['output_root'])/identifier; report=output/'full_native_source_replay.json'
                if report.exists():
                    evidence=replay.native.read(report)
                    if (not evidence['passed'] or evidence['canonical_job']!=job or evidence['new_optimizer_updates_executed']!=0
                            or evidence.get('synthetic_operator_only_not_benchmark_performance',False)):
                        raise ValueError('Only complete original-source independent replays accepted')
                    state['verified'][identifier]=dict(report=str(report),sha256=replay.native.sha(report)); continue
                if identifier in state['failures'] or not complete(job): continue
                if output.exists(): state['failures'][identifier]='incomplete_replay_attempt_preserved'; continue
                handle=ready_slot(job)
                try:
                    command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_shcl_native_replay_job_v1',
                        '--config',args.config,'--job-id',identifier]
                    logs=base/'logs'; logs.mkdir(exist_ok=True)
                    with (logs/(identifier+'.stdout.log')).open('xb') as out,(logs/(identifier+'.stderr.log')).open('xb') as err:
                        child=subprocess.Popen(command,cwd=replay.native.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                                CUDA_VISIBLE_DEVICES=config['gpu_settings']['gpu_uuid'] if job['mode']=='test' else '-1'),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        update(dict(state='replaying_full_original_source_command',active_job=identifier,active_pid=child.pid,
                            active_create_time=psutil.Process(child.pid).create_time(),worker_command=command))
                        gpu_abort={}
                        def guarded_update(values):
                            gpu=gpu_snapshot(config['gpu_settings']) if job['mode']=='test' else {}
                            if gpu and gpu['gpu_free_bytes']<config['gpu_settings']['emergency_gpu_free_bytes']:
                                gpu_abort.update(resource_guard_aborted=True,abort_gpu_snapshot=gpu); api['terminate_tree'](child)
                            update(dict(values,**gpu))
                        code,usage=capped_wait(child,config,api,guarded_update)
                    passed=code==0 and not usage['resource_guard_aborted'] and not gpu_abort and report.exists()
                    write(base/'receipts'/(identifier+'.json'),dict(usage,**gpu_abort,exit_code=code,passed=passed))
                    if passed: state['verified'][identifier]=dict(report=str(report),sha256=replay.native.sha(report))
                    else: state['failures'][identifier]='full_source_replay_failed_preserved'
                    update(dict(active_pid=None,active_job=None))
                finally: handle.close()
            update(dict(state='all413_native_source_replays_verified_TAB_and_paper_backbone_gates_pending' if len(state['verified'])==413
                else 'waiting_for_full_original_training_and_test_results',verified_count=len(state['verified']),
                required_count=413,pending_count=413-len(state['verified']),failed_count=len(state['failures'])))
            if len(state['verified'])==413: return
            time.sleep(30)
    finally: owner.close()


if __name__=='__main__':main()
