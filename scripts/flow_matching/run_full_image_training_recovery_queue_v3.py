"""Preserve all20FM+5WT original full training tasks behind native capacity proof."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts,validate_job
from scripts.flow_matching.wt_full_image_training_contract_v1 import validate_wt_job
from scripts.flow_matching.run_flow_mismatching_full_training_queue_v1 import queue_api
from scripts.flow_matching.run_tccm_full_image_evaluation_queue_v1 import full_gpu_slot
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
from scripts.flow_matching.run_full_image_training_execution_recovery_v3 import verify_capacity


def verify_registration(config):
    verify_receipts(config['source_receipts']);verify_receipts(config['original_failure_receipts'])
    scopes={}
    registered={}
    for objective in ('FM','WT'):
        original=config['original_runtimes'][objective]
        runtime=read(ROOT/original['path']);verify_receipts(runtime['source_receipts'])
        if (runtime['queue']!=original['queue'] or runtime['settings']!=config['settings']
                or runtime['resource_lock']!=config['resource_lock']):
            raise ValueError('Keep the original shared GPU/memory resource guards')
        queue=read(ROOT/runtime['queue']);verify_receipts(queue['source_receipts'])
        scopes[objective]=set()
        for entry in queue['jobs']:
            verify_receipts([entry])
            job=read(ROOT/entry['path']);verify_receipts(job['source_receipts'])
            validator=validate_wt_job if objective=='WT' else validate_job
            validator(job,read(ROOT/job['model_configuration']),read(ROOT/job['optimizer_configuration']),read(ROOT/job['dataset_configuration']))
            scopes[objective].add((job['dataset'],job['variant'],job['seed']))
            if entry['path'] in registered:raise ValueError('Distinct original canonical jobs are required')
            registered[entry['path']]=objective
    if (scopes['FM']!={(d,v,s) for d in ('mvtec','visa') for v in ('conditional','pooled') for s in range(1103,1108)}
            or scopes['WT']!={('mvtec','conditional',s) for s in range(1103,1108)}
            or len(config['ordered_original_jobs'])!=25 or set(config['ordered_original_jobs'])!=set(registered)):
        raise ValueError('The entire original20FM/5WT dataset, ablation and five-seed scope is required')
    expected_disk=sum(read(ROOT/config['original_runtimes'][o]['path'])['full_cohort_required_free_disk_bytes'] for o in ('FM','WT'))
    if config['full_cohort_required_free_disk_bytes']!=expected_disk:
        raise ValueError('Retain conservative disk capacity for every full original checkpoint')
    return registered


def bind_capacity(config):
    capacity=read(ROOT/config['capacity_configuration']);verify_receipts(capacity['source_receipts'])
    state_path=ROOT/config['capacity_state_root']/'status.json'
    terminal=read(state_path)
    if not terminal.get('complete') or terminal.get('state')!='both_full_native_batch32_cuda_capacity_cases_passed':
        raise ValueError('Both actual full original CUDA capacity cases must terminate successfully')
    cases=[];receipts=[]
    for case in capacity['capacity_cases']:
        location=read(ROOT/case['output_directory']/'capacity_probe_result.json');verify_receipts([location])
        proof_path=Path(location['path']);terminal_path=proof_path.parent/'terminal.json'
        proof=read(proof_path)
        if (proof['configuration_sha256']!=sha(ROOT/config['capacity_configuration'])
                or proof['original_job_sha256']!=sha(ROOT/case['original_training_configuration'])):
            raise ValueError('Actual full capacity must bind the exact registered original job and source')
        receipts.extend([dict(path=str(proof_path),sha256=sha(proof_path)),
            dict(path=str(terminal_path),sha256=sha(terminal_path)),proof['checkpoint']])
        cases.append(dict(objective=case['objective'],proof_path=str(proof_path),terminal_path=str(terminal_path)))
    binding=dict(recovery_configuration_sha256=sha(ROOT/config['configuration_path']),cases=cases,proof_receipts=receipts,
        actual_both_case_capacity_terminal_sha256=sha(state_path),capacity_probe_weights_are_never_training_inputs=True)
    verify_capacity(binding)
    return binding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);registered=verify_registration(config)
    if args.validate_only:
        print(dict(original_full_jobs=25,FM_jobs=20,WT_jobs=5,epochs_per_job=2000,batch_size=32,
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
            full_native_capacity_required_before_training=True,benchmark_performance=False));return
    api=queue_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](ROOT/config['lane_lock'])
    if lane is None:raise RuntimeError('Full original image recovery coordinator already owns this lane')
    process=psutil.Process();state=dict(pid=process.pid,create_time=process.create_time(),active_pid=None,
        required_original_jobs=25,completed_original_jobs=0,complete=False,benchmark_performance=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while True:
            capacity_state=read(ROOT/config['capacity_state_root']/'status.json')
            if capacity_state.get('complete'):break
            if capacity_state.get('state')=='failed_capacity_attempt_preserved':
                update(dict(state='failed_capacity_dependency_preserved',active_pid=None));return
            from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
            handle=process_identity(capacity_state.get('pid'),config['capacity_controller_create_time'],'run_full_image_cuda_capacity_queue_v2')
            if not handle['live'] or not handle['create_time_verified']:
                update(dict(state='missing_or_unverified_capacity_dependency',capacity_controller=handle));return
            update(dict(state='waiting_for_both_actual_full_native_cuda_capacity_cases',capacity_controller=handle,
                capacity_state=capacity_state['state']));time.sleep(10)
        binding=bind_capacity(config);binding_path=base/'actual_full_native_capacity_binding.json'
        if binding_path.exists():raise FileExistsError('Preserve every original capacity dispatch binding')
        write(binding_path,binding)
        completed=set()
        first=read(ROOT/config['ordered_original_jobs'][0]);dataset=read(ROOT/first['dataset_configuration'])
        disk_root=Path(dataset['array_receipts'][0]['path']).parent
        while shutil.disk_usage(disk_root).free<config['full_cohort_required_free_disk_bytes']:
            update(dict(state='waiting_for_all25_full_checkpoint_disk_capacity',disk_free_bytes=shutil.disk_usage(disk_root).free));time.sleep(10)
        for job_path in config['ordered_original_jobs']:
            objective=registered[job_path];job=read(ROOT/job_path)
            if objective=='FM':
                from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import completion_valid
            else:
                from scripts.flow_matching.wt_full_image_training_contract_v1 import completion_valid
            output=ROOT/job['output_directory'];final=output/'training_completed.json'
            if final.exists():completion_valid(job,read(final));completed.add(job_path);continue
            update(dict(state='queued_original_full_training',job_id=job['id'],objective=objective,active_pid=None))
            with full_gpu_slot(api,config,update):
                verify_registration(config);verify_capacity(binding)
                attempt=output/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
                command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_full_image_training_execution_recovery_v3',
                    '--recovery-config',args.config,'--config',job_path,'--attempt-directory',str(attempt.resolve()),'--capacity-binding',str(binding_path)]
                with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONUTF8='1',PYTHONHASHSEED=str(job['seed']),
                            CUDA_VISIBLE_DEVICES=config['settings']['gpu_uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    worker=psutil.Process(child.pid)
                    update(dict(state='running_original_full2000epoch_job',active_pid=child.pid,active_create_time=worker.create_time(),actual_command=worker.cmdline()))
                    code,usage=api['guarded_gpu_wait'](child,config['settings'],api,update)
                terminal=dict(returncode=code,worker_pid=child.pid,worker_create_time=state['active_create_time'],command=command,resource_usage=usage)
                write(attempt/'terminal.json',terminal)
                update(dict(active_pid=None,last_terminal=terminal,shared_lock_owned=False))
                if code!=0 or usage['resource_guard_aborted']:
                    update(dict(state='failed_full_training_attempt_preserved',complete=False));return
                completion_valid(job,read(final));completed.add(job_path)
            update(dict(completed_original_jobs=len(completed),state='between_full_original_jobs',active_pid=None));time.sleep(2)
        update(dict(state='all25_full_original_training_jobs_completed',completed_original_jobs=25,complete=True,
            benchmark_performance=False,all_original_experiments_complete=False))
    finally:lane.close()


if __name__=='__main__':main()
