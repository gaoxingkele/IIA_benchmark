"""Guard complete real-data SHCL preprocessing and all52 native loader audits."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess

from scripts.flow_matching.prepare_shcl_released_data_v1 import native,verify
from scripts.flow_matching.timegan_native_common_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True); args=parser.parse_args()
    config=native.read(native.ROOT/args.config); verify(config)
    if Path(config['view_root']).exists(): raise FileExistsError('Preserve complete preprocessing attempts')
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api(); api['resource_snapshot']=resource_snapshot
    base=native.ROOT/config['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Same complete real-data preparation already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=psutil.Process().cmdline(),
        state='starting',active_pid=None,configuration_sha256=native.sha(native.ROOT/args.config))
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json',state)
    try:
        update({})
        with api['resource_slot'](native.ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.prepare_shcl_released_data_v1',
                '--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=native.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='preparing_full_original_data_views',active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(),worker_command=command))
                code,usage=capped_wait(child,config,api,update)
        report_path=Path(config['view_root'])/'full_data_preparation_audit.json'
        passed=code==0 and not usage['resource_guard_aborted'] and report_path.exists()
        if passed:
            report=native.read(report_path)
            if not report['passed'] or report['full_native_loader_modes_audited']!=52:
                raise ValueError('All five full datasets and52 native loaders required')
        write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,
            report_sha256=native.sha(report_path) if report_path.exists() else None,original_training_executed=False))
        update(dict(state='completed_full_data_preparation' if passed else 'failed_preparation_preserved',active_pid=None))
    finally: owner.close()


if __name__=='__main__':main()
