"""Run the exact source numerical diagnostics under original PSM CPU guards."""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import replay_crossad_registered_model_v3 as full
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
from scripts.flow_matching.replay_crossad_full_native_model_v1 import capped_wait


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=full.read(args.config)
    for pin in config['source_receipts']:
        if full.digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen numerical diagnostic source changed')
    old=full.read(config['original_full_replay_configuration'])
    base=full.ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    output=full.ROOT/config['output_directory'];output.mkdir(parents=True,exist_ok=True)
    if (base/'stdout.log').exists():raise FileExistsError('Preserve every original diagnostic attempt')
    api=resource_api();api['resource_snapshot']=resource_snapshot
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('This exact diagnostic already has an owner')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None,benchmark_performance=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        with api['resource_slot'](full.ROOT/old['resource_lock'],old['settings'],update):
            xml=output/'native_attention_lifetime_diagnostic_validation.xml'
            command=[old['python'],'-X','utf8','-m','pytest',*config['test_files'],'-q','-p','no:cacheprovider','--junitxml='+str(xml)]
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=full.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(full.ROOT)+os.pathsep+str(full.ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                identity=psutil.Process(child.pid)
                update(dict(state='running_exact_source_attention_diagnostics',active_pid=child.pid,active_create_time=identity.create_time(),actual_command=identity.cmdline()))
                code,usage=capped_wait(child,old,api,update)
        suite=ET.parse(xml).getroot().find('testsuite') if xml.exists() else None
        checks=int(suite.get('tests','0')) if suite is not None else 0
        valid=bool(code==0 and not usage['resource_guard_aborted'] and checks==config['required_checks']
            and all(suite.get(key)=='0' for key in ('errors','failures','skipped')))
        receipt=dict(exit_code=code,passed=valid,**usage,diagnostic_checks=checks,
            original_replay_settings=old['settings'],diagnostics_are_not_original_full_PSM_replay=True,
            benchmark_performance=False,configuration=args.config,configuration_sha256=full.digest(args.config))
        write(base/'receipt.json',receipt);write(output/'actual_guarded_native_attention_diagnostic_receipt.json',receipt)
        update(dict(state='completed_diagnostics' if valid else 'failed_diagnostics_preserved',active_pid=None))
        if not valid:raise RuntimeError('The actual numerical diagnostic failed; preserve it and do not dispatch replay')
    finally:owner.close()


if __name__=='__main__':main()
