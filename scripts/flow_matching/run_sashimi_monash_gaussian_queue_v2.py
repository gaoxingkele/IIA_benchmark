"""Execute every full Gaussian SaShiMi reconstruction behind measured resource gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.run_sashimi_monash_gaussian_v2 import ROOT,sha,read,verify,verify_result,budget
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.giflow_native_protocol import write_json


def proof_ready(queue,name):
    path=ROOT/queue[name+'_output']
    if not path.exists():return False
    proof=read(path);repair=read(ROOT/queue['cauchy_correction_config'])
    if (not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256']!=sha(ROOT/queue['queue_path'])
        or not proof.get('backend_correction',{}).get('algorithmic_correction')
        or proof['backend_correction']['official_source_sha256']!=repair['official_cauchy_source_sha256']):
        raise ValueError('Full SaShiMi diagnostic source/backend binding differs')
    if name=='data_audit':
        if proof['cases']!=40 or {r['id'] for r in proof['records']}!={j['id'] for j in queue['jobs']}:
            raise ValueError('All40 full SaShiMi data cases required')
        if any(not r['full_original_normalization_bitwise'] or not r['trajectory_split_disjoint'] for r in proof['records']):
            raise ValueError('Complete original preprocessing/split proof missing')
    elif name=='integration':
        if len(proof['cases'])!=2 or {c['profile'] for c in proof['cases']}!=set(queue['profiles']):
            raise ValueError('Both full SaShiMi architecture diagnostics required')
        if any(not c['causality_passed'] or not c['gaussian_nll_backward_passed']
            or c['actual_optimizer_updates']!=1 or c['diagnostic_length']!=728
            or c['generated_shape']!=[2,728,1] for c in proof['cases']):
            raise ValueError('Full-length real FRED Gaussian AR diagnostic incomplete')
        if any(c['full_native_cpu_evaluators']['classifier_updates']!=100
            or c['full_native_cpu_evaluators']['predictor_updates']!=100
            or c['full_native_cpu_evaluators']['generated_shape']!=[22,728,1] for c in proof['cases']):
            raise ValueError('Complete100-epoch native CPU evaluators required')
    else:
        expected={(d,p) for d in queue['datasets'] for p in queue['profiles']}
        if len(proof['cases'])!=8 or {(c['dataset'],c['profile']) for c in proof['cases']}!=expected:
            raise ValueError('All8 full SaShiMi GPU/evaluator cases required')
        for case in proof['cases']:
            config=next(read(ROOT/j['model_config']) for j in queue['jobs']
                if read(ROOT/j['model_config'])['dataset']==case['dataset']
                and read(ROOT/j['model_config'])['profile']==case['profile'])
            full=budget(config)
            if (not case['passed'] or case['full_epoch_updates']!=full['batches']
                or case['full_train_samples']!=config['train_samples'] or case['full_test_samples']!=config['test_samples']
                or case['length']!=config['length'] or case['classifier_updates']!=full['classifier_updates']
                or case['predictor_updates']!=full['predictor_updates']):
                raise ValueError('Unabridged SaShiMi training/evaluator capacity proof required')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True)
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full SaShiMi controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),queue_sha256=sha(ROOT/args.queue),
        state='starting',active_pid=None,active_job=None,jobs={})
    def update(value):
        state.update(value);write_json(base/'status.json',state)
    def launch(flag,name,cpu=False,job=None):
        command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_sashimi_monash_gaussian_v2',
            '--queue',args.queue]+(['--job-id',job['id']] if job else [flag])
        logs=base/'logs';logs.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='-1' if cpu else '0')
        with (logs/(name+'.stdout.log')).open('xb') as out,(logs/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(dict(state='running_diagnostic' if cpu else 'running_full_gpu_pipeline',
                active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
            return capped_wait(child,dict(settings=queue['data_audit_settings']),api,update) if cpu else api['guarded_gpu_wait'](child,queue['settings'],api,update)
    try:
        for name,flag,cpu in [('integration','--integration-check',True),('data_audit','--audit-data',True),('gpu_preflight','--gpu-preflight',False)]:
            if proof_ready(queue,name):continue
            if (base/'logs'/(name+'.stdout.log')).exists():raise FileExistsError('Preserve prior SaShiMi diagnostic attempt')
            slot=api['resource_slot'](ROOT/queue['data_audit_lock'],queue['data_audit_settings'],update) if cpu else full_gpu_memory_slot(api,queue,update)
            with slot:code,peaks=launch(flag,name,cpu)
            valid=code==0 and proof_ready(queue,name)
            write_json(base/(name+'_resource_receipt.json'),dict(exit_code=code,passed=valid,**peaks))
            update(dict(active_pid=None))
            if not valid:raise RuntimeError('SaShiMi diagnostic failed and preserved')
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
            update(dict(state='between_full_jobs',active_pid=None,active_job=None));time.sleep(queue['settings']['controller_yield_seconds'])
        update(dict(state='completed' if len(state['jobs'])==40 and all(v in ('completed','completed_preexisting') for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'))
    finally:owner.close()


if __name__=='__main__':main()
