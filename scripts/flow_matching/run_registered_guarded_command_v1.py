"""Execute a frozen full diagnostic command under the registered resource lock."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT,resource_api
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import sha
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for entry in config['source_receipts']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Frozen diagnostic source changed')
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'stdout.log').exists():raise FileExistsError('Previous diagnostic attempt must remain intact')
    api=resource_api();lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This exact diagnostic already has a controller')
    state=dict(pid=os.getpid(),config_sha256=sha(ROOT/args.config),state='starting',active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                    PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2')
                child=subprocess.Popen(config['command'],cwd=ROOT,stdout=out,stderr=err,env=environment,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid)
                update(dict(state='running',active_pid=child.pid,active_create_time=handle.create_time(),
                    actual_command=handle.cmdline()))
                code,peaks=api['guarded_wait'](child,config['settings'],update)
        output=ROOT/config['expected_output'];proof=json.loads(output.read_text(encoding='utf-8')) if output.exists() else None
        valid=bool(code==0 and proof and proof.get('passed') and proof.get('diagnostic_only')
            and proof.get('queue_sha256')==config['expected_queue_fingerprint'])
        write(base/'resource_receipt.json',dict(exitcode=code,full_diagnostic_passed=valid,**peaks))
        update(dict(state='completed' if valid else 'failed_or_partial_preserved',active_pid=None,
            full_diagnostic_passed=valid,output_sha256=sha(output) if output.exists() else None))
        if not valid:raise RuntimeError('Full registered diagnostic did not pass; preserve all logs')
    finally:lane.close()


if __name__=='__main__':main()
