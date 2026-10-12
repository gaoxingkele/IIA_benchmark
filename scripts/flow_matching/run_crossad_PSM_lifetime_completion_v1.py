"""Finish the preserved full PSM release review without changing any guard.

Actual numerical lifetime diagnostics precede full native replay and the
unchanged independent post-review. Failed attempts are never overwritten or
silently retried. Existing whole139 controllers remain live and byte-identical.
"""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import time

import psutil

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write


def verify(config):
    for pin in config['source_receipts']:
        if full.digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen PSM complete-replay binding changed')
    base=full.read(config['original_completion_configuration'])
    canonical_jobs,_,covered=full.verify(base)
    canonical=canonical_jobs[config['canonical_id']]
    native=full.read(config['native_configuration']);post=full.read(config['post_configuration'])
    queue=full.read(native['queue']);job=next(j for j in queue['jobs'] if j['id']==native['job_id'])
    from scripts.flow_matching.run_crossad_recovery_completion_handoff_v1 import exact_route
    route=config['preserved_recovery_route']
    if exact_route(route,canonical,queue,full.read(base['original_queue']))!=job:
        raise ValueError('Only the same exact already completed recovery can be reviewed')
    if (config['required_original_jobs']!=139 or config['original_scope']!=full.SCOPE
            or any(native[k]!=base[k] or post[k]!=base[k] for k in full.GUARDS)
            or config['score_tolerance']!=full.SCORE_TOLERANCE or config['metric_tolerance']!=full.METRIC_TOLERANCE
            or post['native_configuration']!=config['native_configuration']):
        raise ValueError('Entire original139 scope and all original numerical guards are required')
    point=full.read(native['independent_point_completion']);point_terminal=full.read(config['point_terminal'])
    result=full.read(full.ROOT/job['output_directory']/'result.json')
    if (not point['passed'] or point['original_source_result']!=result or point['job_id']!=job['id']
            or point_terminal['exit_code']!=0 or point_terminal['resource_guard_aborted']
            or not point_terminal['independent_stage_passed'] or not point['full_raw_labels_and_source_rows_checked']):
        raise ValueError('Reuse only the actually passed complete original raw point certificate')
    return native,post,job,covered


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=full.read(args.config);native,post,job,covered=verify(config)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.replay_crossad_full_native_model_v1 import capped_wait
    api=resource_api();api['resource_snapshot']=resource_snapshot
    base=full.ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The exact PSM lifetime recovery already has a live owner')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None,
        required_original_jobs=139,original_scope=full.SCOPE,canonical_id=config['canonical_id'],
        original_fixed_release_training_seed_slots=0,all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        diagnostic_path=full.ROOT/config['diagnostic_terminal']
        while not diagnostic_path.exists():
            expected=config['diagnostic_owner'];live=psutil.Process(expected['pid'])
            if live.create_time()!=expected['create_time'] or live.cmdline()!=expected['command']:
                raise ValueError('The diagnostic owner identity changed; do not restart by inference')
            update(dict(state='waiting_actual_guarded_native_lifetime_diagnostics',actual_verified_diagnostic_owner=expected))
            time.sleep(config['poll_seconds'])
        diagnostic=full.read(diagnostic_path)
        if diagnostic['exit_code']!=0 or not diagnostic['passed'] or diagnostic['resource_guard_aborted']:
            raise ValueError('Preserve failed exact lifetime diagnostics; no full-source retry is admitted')
        for name,spec,path,module,proof_name,receipt_key in [
            ('native',native,config['native_configuration'],'scripts.flow_matching.replay_crossad_residual_lifetime_v4',
             'full_native_model_replay.json','full_native_model_replay_passed'),
            ('post',post,config['post_configuration'],'scripts.flow_matching.run_crossad_completion_replay_queue_v1',
             'independent_completed_replay_review.json','independent_stage_passed')]:
            verify(config)
            stage_base=full.ROOT/spec['state_root'];stage_base.mkdir(parents=True,exist_ok=True)
            lock=api['exclusive_lock'](stage_base/'controller.lock')
            if lock is None:raise RuntimeError('An existing numerical stage is owned; never duplicate it')
            try:
                target=full.ROOT/spec['artifact_directory' if name=='native' else 'output_directory']
                if target.exists():raise FileExistsError('Preserve every complete or failed native stage')
                def stage_update(values):
                    update(dict(values,stage=name));write(stage_base/'status.json',state)
                with api['resource_slot'](full.ROOT/spec['resource_lock'],spec['settings'],stage_update):
                    command=[spec['python'],'-X','utf8','-u','-m',module,'--config',path,
                             '--worker' if name=='native' else '--stage-worker']
                    with (stage_base/'stdout.log').open('xb') as out,(stage_base/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=full.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONPATH=str(full.ROOT)+os.pathsep+str(full.ROOT/'src'),
                                PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                                OMP_NUM_THREADS=str(spec['cpu_threads']),MKL_NUM_THREADS=str(spec['cpu_threads']),OPENBLAS_NUM_THREADS=str(spec['cpu_threads'])),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        identity=psutil.Process(child.pid)
                        stage_update(dict(state='executing_entire_original_PSM_'+name,active_pid=child.pid,
                            active_create_time=identity.create_time(),actual_command=identity.cmdline()))
                        code,usage=capped_wait(child,spec,api,stage_update)
                proof=target/proof_name
                passed=code==0 and not usage['resource_guard_aborted'] and proof.exists()
                if passed:
                    passed=full.read(proof).get('passed',True)
                write(stage_base/'receipt.json',dict(exit_code=code,**usage,**{receipt_key:passed}))
                stage_update(dict(state='completed_'+name if passed else 'failed_'+name+'_preserved',active_pid=None))
                if not passed:return
                if name=='native':full.native_terminal(native,job)
            finally:lock.close()
        from scripts.flow_matching.run_crossad_recovery_completion_handoff_v1 import completed_review
        certificate=completed_review(config['post_configuration'],job,full.read(config['original_completion_configuration']))
        write(base/'full_release_completion_certificate.json',dict(canonical_id=config['canonical_id'],
            actual_recovery_job=job,certificate=certificate,additional_training_seed_slots=0,
            original_fixed_release_training_seed_slots=0,required_original_jobs=139,all_original_experiments_complete=False))
        update(dict(state='actual_entire_PSM_native_and_independent_replay_verified',actual_complete_release_replay=True,active_pid=None))
    except Exception as error:
        update(dict(state='failure_preserved',exception_type=type(error).__name__,exception=str(error),active_pid=None))
        raise
    finally:owner.close()


if __name__=='__main__':main()
