"""Full canonical LS4 corrected attempts; never reduce original resource gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.run_ls4_cauchy_corrected_v2 import ROOT,sha,verify,verify_result
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.giflow_native_protocol import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def proof_ready(queue,name):
    path=ROOT/queue[name+'_output']
    if not path.exists():return False
    proof=read(path);correction=proof.get('backend_correction',{})
    if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=sha(ROOT/queue['queue_path']) or not correction.get('algorithmic_correction'):
        raise ValueError('Full corrected LS4 proof differs')
    if name=='data_audit':
        if proof['cases']!=20:raise ValueError('All20 full corrected LS4 data audits required')
    else:
        expected={(d['dataset'] if isinstance(d,dict) else d,p)
                  for d in queue['datasets'] for p in ('released','paper_literal')}
        if {(c['dataset'],c['protocol']) for c in proof['cases']}!=expected or len(proof['cases'])!=4:
            raise ValueError('All4 full corrected LS4 GPU/evaluator cases required')
        if any(c['original_classifier_epochs']!=100 or c['original_predictor_epochs']!=100
               or not c['complete_metric_interface_passed'] for c in proof['cases']):
            raise ValueError('Full corrected LS4 original evaluators required')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True)
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Corrected full LS4 controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),queue_sha256=sha(ROOT/args.queue),state='starting',active_pid=None,active_job=None,jobs={})
    def update(value):
        state.update(value);write_json(base/'status.json',state)
    def launch(flag,name,cpu=False,job=None):
        command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_ls4_cauchy_corrected_v2',
            '--queue',args.queue]+(['--job-id',job['id']] if job else [flag])
        logs=base/'logs';logs.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='-1' if cpu else '0')
        with (logs/(name+'.stdout.log')).open('xb') as out,(logs/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(dict(state='running_full_data_audit' if cpu else 'running_full_gpu_pipeline',active_pid=child.pid,
                active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
            return capped_wait(child,dict(settings=queue['data_audit_settings']),api,update) if cpu else api['guarded_gpu_wait'](child,queue['settings'],api,update)
    try:
        for name,flag,cpu in [('data_audit','--audit-data',True),('gpu_preflight','--gpu-preflight',False)]:
            if proof_ready(queue,name):continue
            if (base/'logs'/(name+'.stdout.log')).exists():raise FileExistsError('Preserve prior full corrected diagnostic attempt')
            slot=api['resource_slot'](ROOT/queue['data_audit_lock'],queue['data_audit_settings'],update) if cpu else full_gpu_memory_slot(api,queue,update)
            with slot:code,peaks=launch(flag,name,cpu)
            valid=code==0 and proof_ready(queue,name)
            write_json(base/(name+'_resource_receipt.json'),dict(exit_code=code,passed=valid,**peaks))
            update(dict(active_pid=None))
            if not valid:raise RuntimeError('Full corrected LS4 diagnostic failed and preserved')
        for job in queue['jobs']:
            if (ROOT/job['output_directory']/'result.json').exists():
                verify_result(queue,job);state['jobs'][job['id']]='completed_preexisting';continue
            if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                state['jobs'][job['id']]='failed_or_partial_preserved';update({});continue
            update(dict(active_job=job['id']))
            with full_gpu_memory_slot(api,queue,update):code,peaks=launch('',job['id'],job=job)
            valid=False
            if code==0 and (ROOT/job['output_directory']/'result.json').exists():verify_result(queue,job);valid=True
            state['jobs'][job['id']]='completed' if valid else 'failed_or_partial_preserved'
            write_json(base/'receipts'/(job['id']+'.json'),dict(exit_code=code,full_result_validated=valid,**peaks))
            update(dict(state='between_original_jobs',active_pid=None,active_job=None));time.sleep(queue['settings']['controller_yield_seconds'])
        update(dict(state='completed' if len(state['jobs'])==20 and all(v in ('completed','completed_preexisting') for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'))
    finally:owner.close()


if __name__=='__main__':main()
