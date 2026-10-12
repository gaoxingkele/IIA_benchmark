"""Keep all existing read-only caps for full native Torch1.11 operator/data audit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.timegan_native_common_v1 import write


def verify(config):
    for pin in config['source_receipts']:
        if native.sha(native.ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen native environment audit changed')
    actual=json.loads(subprocess.check_output([config['python'],'-X','utf8','-m',
        'scripts.flow_matching.shcl_author_audits_v1','--environment-only'],cwd=native.ROOT,encoding='utf-8',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
    if actual!=config['environment']: raise ValueError('Actual native numerical packages differ from the registered environment')
    from scripts.flow_matching.shcl_author_audits_v1 import compiled
    _,verification=compiled(config['audit_kind'])
    # Actual native metadata was just read in the target process. Pure host
    # registration checks can bind that verified value without importing Torch.
    from types import FunctionType
    bindings=dict(verification.__globals__,environment=lambda:actual,data_environment=lambda:actual)
    FunctionType(verification.__code__,bindings)(config)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); config=native.read(native.ROOT/args.config); verify(config)
    output=Path(config['view_root']) if config['audit_kind']=='data' else native.ROOT/config['output_root']
    if output.exists(): raise FileExistsError('Preserve every full native audit attempt')
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api(); api['resource_snapshot']=resource_snapshot
    base=native.ROOT/config['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Same full native environment audit already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),
        configuration_sha256=native.sha(native.ROOT/args.config),state='starting',active_pid=None)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json',state)
    try:
        update({})
        with api['resource_slot'](native.ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.shcl_author_audits_v1','--config',args.config]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=native.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='running_full_native_audit',active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(),worker_command=command))
                code,usage=capped_wait(child,config,api,update)
        report=output/('full_released_operator_audit.json' if config['audit_kind']=='operators' else 'full_data_preparation_audit.json')
        passed=code==0 and not usage['resource_guard_aborted'] and report.exists()
        if passed:
            values=native.read(report)
            if not values['passed'] or (values.get('full_operator_profiles')!=13 if config['audit_kind']=='operators'
                    else values['full_native_loader_modes_audited']!=52):
                raise ValueError('Every complete native obligation required')
        write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,
            report_sha256=native.sha(report) if report.exists() else None,original_training_executed=False))
        update(dict(state='completed_full_native_audit' if passed else 'failed_native_audit_preserved',active_pid=None))
    finally: owner.close()


if __name__=='__main__':main()
