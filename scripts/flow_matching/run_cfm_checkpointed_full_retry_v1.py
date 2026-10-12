"""Admit the same original full400 CFM job with its unchanged CPU family guard."""
import argparse
from datetime import datetime,timezone
import os
import subprocess

from scripts.flow_matching.run_cfm_checkpointed_original_job_v1 import original,verify_registration


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);args=parser.parse_args()
    config=original.read(original.ROOT/args.config);queue,job=verify_registration(config)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api=resource_api();api['resource_snapshot']=resource_snapshot
    base=original.ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This full original checkpointed retry already has a live owner')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        canonical_job_id=config['canonical_job']['id'],active_pid=None,original_required_epochs=400,
        retained_partial_epochs_credited=0,original_canonical_CFM_jobs=285,
        additional_training_seed_slots=0,all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());original.write(base/'status.json',state)
    try:
        if (original.ROOT/job['output_directory']).exists() or (original.ROOT/job['artifact_directory']).exists():
            raise FileExistsError('Preserve all scientific results and incomplete attempts')
        with api['resource_slot'](original.ROOT/config['family_lock'],config['resource_policy'],update):
            verify_registration(config)
            command=[str((original.ROOT/queue['python']).resolve()),'-X','utf8','-u','-m',
                'scripts.flow_matching.run_cfm_checkpointed_original_job_v1','--config',args.config]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=original.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(original.ROOT)+os.pathsep+str(original.ROOT/'src'),
                        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                identity=psutil.Process(child.pid)
                update(dict(state='running_full_original400_with_complete_epoch_checkpoints',active_pid=child.pid,
                    active_create_time=identity.create_time(),actual_command=identity.cmdline()))
                code,usage=api['guarded_wait'](child,config['resource_policy'],update)
        # The original worker validates its full model/predictions/MSE after all
        #400 epochs; avoid importing numerical packages in the waiting parent.
        result_path=original.ROOT/job['output_directory']/'result.json'
        result=original.read(result_path) if result_path.exists() else None
        passed=(code==0 and not usage['resource_guard_aborted'] and result is not None
            and result['status']=='completed' and result['epochs_completed']==400
            and result['optimizer_updates']==400 and result['job_sha256']==original.fingerprint(job))
        original.write(base/'receipt.json',dict(exit_code=code,complete=passed,
            full_original400_checkpointed_result_verified=passed,resource_observations=usage))
        update(dict(state='completed' if passed else 'failed_attempt_preserved',active_pid=None,
            full_original400_checkpointed_result_verified=passed))
    finally:lane.close()


if __name__=='__main__':main()
