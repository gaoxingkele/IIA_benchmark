"""Bounded full-native checkpoint/optimizer/RNG audit before stage recovery.

This executes the original complete data and model audit, not benchmark training.
It shares the existing read-only CPU lock and preserves every original attempt.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.timegan_stage_resume_v2 import binding
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    runtime=read(ROOT/args.runtime_config)
    for source in runtime['source_receipts']:
        if sha(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen original stage-audit resource differs: '+source['path'])
    audit_config=read(ROOT/runtime['audit_configuration'])
    queue,model,job,records=binding(audit_config)
    if (runtime['python']!=queue['python'] or runtime['resource_lock']!='experiments/runs/fm_readonly_checkpoint_audit_cpu.lock'
            or runtime['settings']['emergency_commit_headroom_bytes']<16*1024**3
            or runtime['settings']['minimum_commit_headroom_bytes']<runtime['settings']['emergency_commit_headroom_bytes']
                +runtime['settings']['maximum_worker_private_bytes']+3*1024**3):
        raise ValueError('Original interpreter, existing read-only mutex and complete memory reserve required')
    if args.probe_only:
        print(json.dumps(dict(canonical_id=job['id'],original_full_scope_jobs=len(queue['jobs']),
            restored_complete_pretraining_updates=records[-1]['optimizer_updates'],
            tensorflow_imported='tensorflow' in sys.modules,audit_not_benchmark_performance=True)))
        return
    output=ROOT/audit_config['native_restore_audit']
    if output.exists():
        raise FileExistsError('Preserve completed or partial native restore audit')
    base=ROOT/runtime['state_root']
    base.mkdir(parents=True,exist_ok=True)
    api=resource_api()
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Same restore audit already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None,
        runtime_sha256=sha(ROOT/args.runtime_config),canonical_id=job['id'])
    def update(values):
        state.update(values)
        state['captured_utc']=datetime.now(timezone.utc).isoformat()
        write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/runtime['resource_lock'],runtime['settings'],update):
            bootstrap=("import sys,runpy;sys.path.insert(0,"+repr(str(ROOT))+");"
                "sys.argv=['native_full_stage_audit','--config',"+repr(runtime['audit_configuration'])+"];"
                "runpy.run_module('scripts.flow_matching.audit_timegan_stage_resume_v2',run_name='__main__')")
            command=[runtime['python'],'-X','utf8','-u','-c',bootstrap]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg',
                        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='running_full_original_checkpoint_restore_and_native_transition_audit',
                    active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,runtime,api,update)
        passed=False
        if code==0:
            proof=read(output)
            if (not proof['passed'] or proof['canonical_job_id']!=job['id']
                    or not proof['all_transition_variable_values_bitwise_equal']
                    or not proof['numpy_minibatch_noise_trajectory_identical']
                    or not proof['checkpoint_all_variable_restoration']['all_variable_values_bitwise_equal']
                    or proof['restored_completed_stage_updates']!=records[-1]['optimizer_updates']
                    or proof['full_native_joint_transition_iterations_per_path']!=audit_config['audit_joint_iterations']):
                raise ValueError('Whole original optimizer/model/RNG restoration and native transitions required')
            passed=True
        receipt=dict(exit_code=code,full_native_restore_and_transition_audit_passed=passed,**usage,
            native_audit_path=str(output),native_audit_sha256=sha(output) if output.exists() else None,
            verification_only_not_benchmark_training=True,additional_training_seed_slots=0)
        write(base/'receipt.json',receipt)
        update(dict(state='completed' if passed else 'failed_audit_preserved',active_pid=None))
        print(json.dumps(dict(full_native_restore_and_transition_audit_passed=passed,exit_code=code,
            resource_guard_aborted=usage['resource_guard_aborted'])))
    finally:
        owner.close()


if __name__=='__main__':
    main()
