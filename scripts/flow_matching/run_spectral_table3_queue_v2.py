"""Guard all four full long-series pipelines with case-specific native capacity proofs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_spectral_table3_original_v2 import ROOT,read,sha,verify
from scripts.flow_matching.spectral_table3_bootstrap_v1 import verify_training_coverage,check_evaluator_repeats
from scripts.flow_matching.run_spectral_table2_method_gated_queue_v1 import complete_native_slot
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def full_capacity_proof(queue,job,proof):
    config=read(ROOT/job['model_config']);shape=queue['datasets'][config['dataset']]['shape']
    expected_test=[shape[0]-int(shape[0]*.8),*shape[1:]]
    return bool(proof.get('passed') and proof.get('diagnostic_only')
        and proof.get('queue_sha256')==sha(ROOT/queue['queue_path']) and proof.get('job_id')==job['id']
        and proof.get('full_original_config_sha256')==job['model_config_sha256']
        and proof.get('training_batch_shape')==[config['author_hyperparameters']['batch_size'],*shape[1:]]
        and proof.get('test_shape')==expected_test and proof.get('full_loader_rows')==int(shape[0]*.8)
        and proof.get('num_workers')==4 and len(set(proof.get('worker_pids',[])))==4
        and all(isinstance(p,int) and p>0 for p in proof.get('worker_pids',[]))
        and proof.get('worker_exitcodes')==[0]*4
        and proof.get('native_full_generation_and_ten_repeat_evaluator_passed'))


def complete_result(queue,job):
    output=ROOT/job['output_directory'];path=output/'result.json'
    if not path.exists():return False
    result=read(path);artifact=Path(job['artifact_directory'])
    if (result.get('status')!='completed' or result.get('queue_sha256')!=sha(ROOT/queue['queue_path'])
            or result.get('model_config_sha256')!=job['model_config_sha256']):
        raise ValueError('Full Table3 result identity differs')
    for stage in ('training','evaluation'):
        receipt=artifact/(stage+'_receipt.json')
        if sha(receipt)!=result[stage+'_receipt_sha256']:raise ValueError('Original stage receipt differs')
        state=read(receipt)
        if state['status']!='completed' or state['diagnostic_only']:raise ValueError('Full original stage incomplete')
        c=read(artifact/(stage+'_contract.json'))
        if sha(artifact/(stage+'_contract.json'))!=state['contract_sha256']:raise ValueError('Full original stage contract differs')
        if stage=='training':
            verify_training_coverage(state,c)
            if sha(artifact/'full_final_training.pt')!=state['final_checkpoint_sha256']:
                raise ValueError('Full original final checkpoint differs')
            if sha(state['selected_checkpoint'])!=state['selected_checkpoint_sha256']:
                raise ValueError('Marginal-selected checkpoint differs')
        elif state['optimizer_updates'] or len(state['evaluations'])!=1:
            raise ValueError('Full original standalone test stage missing')
        for record in state['evaluations']:
            if sha(record['arrays'])!=record['arrays_sha256'] or sha(record['evaluator_rng'])!=record['evaluator_rng_sha256']:
                raise ValueError('Full original arrays/evaluator RNG differ')
            check_evaluator_repeats(record['repeats'],record['scores'])
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    queue=read(ROOT/args.queue);verify(queue)
    if args.probe_only:
        print(json.dumps(dict(full_jobs=len(queue['jobs']),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,full_epochs_per_job=1000,full_native_case_preflights_required=True)))
        return
    api=guarded_api();base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('This full Table3 queue already has a live controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),queue_sha256=sha(ROOT/args.queue),
        state='starting',active_pid=None,active_job=None,jobs={})
    def update(values):state.update(values);write(base/'status.json',state)
    environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2')
    def launch(command,identity):
        logs=base/'logs';logs.mkdir(exist_ok=True)
        with (logs/(identity+'.stdout.log')).open('xb') as out,(logs/(identity+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=psutil.Process(child.pid)
            update(dict(state='running',active_job=identity,active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
            code,usage=api['guarded_gpu_wait'](child,queue['settings'],api,update)
        write(base/'receipts'/(identity+'.json'),dict(exit_code=code,command=command,**usage))
        update(dict(state='between_full_stages',active_job=None,active_pid=None))
        return code
    try:
        data=read(ROOT/queue['cpu_data_audit'])
        if not data['passed'] or not data['diagnostic_only'] or data['queue_sha256']!=sha(ROOT/args.queue):
            raise ValueError('Independent full original data audit required')
        for record in data['datasets'].values():
            if sha(record['arrays'])!=record['arrays_sha256']:raise ValueError('Independent normalized full arrays differ')
        for job in queue['jobs']:
            if complete_result(queue,job):state['jobs'][job['id']]='completed_full_original_runner';continue
            if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                state['jobs'][job['id']]='failed_or_partial_preserved';update({});continue
            proof_path=base/'preflight'/(job['id']+'_gpu_v1.json')
            if not proof_path.exists():
                identity='preflight_'+job['id']
                if (base/'logs'/(identity+'.stdout.log')).exists():
                    state['jobs'][job['id']]='awaiting_registered_full_preflight_retry';update({});continue
                command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_table3_original_v2',
                    '--queue',args.queue,'--job-id',job['id'],'--preflight-output',proof_path.relative_to(ROOT).as_posix()]
                with complete_native_slot(api,queue,update):code=launch(command,identity)
                time.sleep(queue['settings']['controller_yield_seconds'])
                if code or not proof_path.exists():
                    state['jobs'][job['id']]='full_native_preflight_failed_preserved';update({});continue
            if not full_capacity_proof(queue,job,read(proof_path)):
                raise ValueError('Full original case capacity proof differs')
            command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.run_spectral_table3_original_v2',
                '--queue',args.queue,'--job-id',job['id']]
            with complete_native_slot(api,queue,update):code=launch(command,job['id'])
            valid=code==0 and complete_result(queue,job)
            state['jobs'][job['id']]='completed_full_original_runner' if valid else 'failed_or_partial_preserved'
            update({});time.sleep(queue['settings']['controller_yield_seconds'])
        update(dict(state='completed_full_original_runner_results' if all(v=='completed_full_original_runner' for v in state['jobs'].values())
            else 'completed_with_unresolved_jobs',independent_trained_model_re_evaluation_required=True))
    finally:owner.close()


if __name__=='__main__':main()
