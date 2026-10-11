"""Queue the same original RCGAN seed's remaining262 full native epochs."""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,write,sha


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen full original recovery registration changed')
    from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1 import verify_registration as verify_state
    state_config=read(ROOT/config['checkpoint_state_configuration'])
    queue,job,_,_,_=verify_state(state_config)
    proof=read(ROOT/state_config['output_directory']/'checkpoint_resume_state_audit.json')
    terminal=read(ROOT/state_config['state_root']/'receipt.json')
    if (config['canonical_job']!=job or not proof['passed'] or terminal['exit_code']!=0
            or terminal['resource_guard_aborted'] or not terminal['checkpoint_resume_state_passed']
            or config['settings']!=queue['settings'] or config['resource_lock']!=queue['resource_lock']
            or config['full_original_epochs']!=500 or config['verified_completed_epochs']!=238
            or config['remaining_full_epochs']!=262 or config['recovery_job']['seed']!=job['seed']):
        raise ValueError('Preserve exact original500epoch seed/family/resource guards after successful state proof')
    if read(ROOT/queue['native_evaluator_execution_receipt'])['exit_code']!=0:
        raise ValueError('Successful full original native evaluator preparation remains required')
    return queue,state_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--proof-binding');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);queue,state_config=verify_registration(config)
    if args.probe_only:
        print(dict(original_epochs=500,verified_prior_epochs=238,remaining_full_epochs=262,
            original_seed=config['canonical_job']['seed'],full_original_seed_slots=len(queue['jobs']),
            tensorflow_imported='tensorflow' in sys.modules,numpy_imported='numpy' in sys.modules));return
    if args.worker:
        from scripts.flow_matching.rcgan_native_training_resume_runtime_v1 import run
        run(config,queue,read(ROOT/args.proof_binding));return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.rcgan_native_training_resume_runtime_v1 import verify_result
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('This exact original-state recovery has a live controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting')
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        output=ROOT/config['recovery_job']['output_directory']
        if output.exists():raise FileExistsError('Preserve all prior failed/partial full recovery attempts')
        binding=dict(configuration_sha256=sha(ROOT/args.config),
            checkpoint_state_proof_sha256=sha(ROOT/state_config['output_directory']/'checkpoint_resume_state_audit.json'),
            checkpoint_state_terminal_sha256=sha(ROOT/state_config['state_root']/'receipt.json'),
            native_evaluator_proof_sha256=sha(ROOT/queue['native_evaluator_proof']),
            native_evaluator_terminal_sha256=sha(ROOT/queue['native_evaluator_execution_receipt']))
        binding_path=base/'full_native_resume_proof_binding.json'
        if binding_path.exists():raise FileExistsError('Preserve original immutable proof binding')
        write(binding_path,binding)
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            verify_registration(config)
            command=[queue['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_rcgan_native_state_recovery_v1',
                '--config',args.config,'--worker','--proof-binding',str(binding_path)]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                        PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                identity=psutil.Process(child.pid)
                update(dict(state='running_original_full262_remaining_epochs',active_pid=child.pid,
                    active_create_time=identity.create_time(),actual_command=identity.cmdline()))
                def running(values):update(dict(values,state='running_original_full262_remaining_epochs'))
                code,usage=capped_wait(child,config,api,running)
        passed=False
        if code==0 and not usage['resource_guard_aborted']:
            verify_result(config,queue);passed=True
        write(base/'receipt.json',dict(exit_code=code,full_original500epoch_resume_verified=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,
            full_original500epoch_resume_verified=passed,all_original_experiments_complete=False))
    finally:owner.close()


if __name__=='__main__':main()
