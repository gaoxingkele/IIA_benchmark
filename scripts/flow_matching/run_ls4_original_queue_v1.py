"""Serial full LS4 jobs with bounded CPU data audit and unchanged full GPU gates."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.run_ls4_original_v1 import ROOT, read, verify, verify_result, sha, write
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def proof_ready(queue, name):
    path=ROOT/queue[name+'_output']
    if not path.exists():
        return False
    proof=read(path)
    if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=sha(ROOT/queue['queue_path']):
        raise ValueError('Frozen complete LS4 diagnostic proof differs')
    if name=='gpu_preflight':
        expected={(d,p) for d in ('fred_md','nn5_daily') for p in ('released','paper_literal')}
        if {(c['dataset'],c['protocol']) for c in proof['cases']}!=expected or len(proof['cases'])!=4:
            raise ValueError('All4 full LS4 model/evaluator cases required')
        if any(c['original_classifier_epochs']!=100 or c['original_predictor_epochs']!=100
               or not c['complete_metric_interface_passed'] for c in proof['cases']):
            raise ValueError('Full original LS4 evaluators not verified')
    elif proof['cases']!=20:
        raise ValueError('All20 complete LS4 split cases required')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    if args.probe_only:
        import sys,json
        print(json.dumps(dict(full_original_jobs=len(queue['jobs']),torch_imported='torch' in sys.modules,
            original_released_epochs=10000,paper_literal_epochs=7000)));return
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Full LS4 controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),queue_sha256=sha(ROOT/args.queue),
        state='starting',active_pid=None,active_job=None,jobs={})
    def update(values):
        state.update(values);write(base/'status.json',state)
    def launch(flag, logname, cpu=False, job=None):
        cmd=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_ls4_original_v1',
             '--queue',args.queue]+(['--job-id',job['id']] if job else [flag])
        logs=base/'logs';logs.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            WANDB_MODE='disabled',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='-1' if cpu else '0')
        with (logs/(logname+'.stdout.log')).open('xb') as out,(logs/(logname+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=psutil.Process(child.pid)
            update(dict(state='running_full_data_audit' if cpu else 'running_full_gpu_pipeline',
                active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
            return capped_wait(child,{'settings':queue['data_audit_settings']},api,update) if cpu else api['guarded_gpu_wait'](child,queue['settings'],api,update)
    try:
        for name,flag,cpu in [('data_audit','--audit-data',True),('gpu_preflight','--gpu-preflight',False)]:
            if args.audit_only and not cpu:
                return
            if proof_ready(queue,name):
                continue
            if (base/'logs'/(name+'.stdout.log')).exists():
                raise FileExistsError('Preserve failed/partial LS4 diagnostic; register new exact attempt')
            slot=api['resource_slot'](ROOT/queue['data_audit_lock'],queue['data_audit_settings'],update) if cpu else full_gpu_memory_slot(api,queue,update)
            with slot:
                code,peaks=launch(flag,name,cpu)
            valid=code==0 and proof_ready(queue,name)
            write(base/(name+'_resource_receipt.json'),dict(exit_code=code,passed=valid,**peaks))
            update(dict(active_pid=None))
            if not valid:
                update(dict(state='failed_diagnostic_preserved'));raise RuntimeError('Full LS4 diagnostic failed')
        for job in queue['jobs']:
            if (ROOT/job['output_directory']/'result.json').exists():
                verify_result(queue,job);state['jobs'][job['id']]='completed_preexisting';continue
            if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                state['jobs'][job['id']]='failed_or_partial_preserved';update({});continue
            update(dict(active_job=job['id']))
            with full_gpu_memory_slot(api,queue,update):
                code,peaks=launch('',job['id'],job=job)
            valid=False
            if code==0 and (ROOT/job['output_directory']/'result.json').exists():
                verify_result(queue,job);valid=True
            state['jobs'][job['id']]='completed' if valid else 'failed_or_partial_preserved'
            write(base/'receipts'/(job['id']+'.json'),dict(exit_code=code,full_result_validated=valid,**peaks))
            update(dict(state='between_original_jobs',active_pid=None,active_job=None))
            time.sleep(queue['settings']['controller_yield_seconds'])
        complete=all(v in ('completed','completed_preexisting') for v in state['jobs'].values())
        update(dict(state='completed' if complete else 'completed_with_unresolved_jobs',all20_original_jobs_completed=complete))
    finally:
        owner.close()


if __name__=='__main__':
    main()
