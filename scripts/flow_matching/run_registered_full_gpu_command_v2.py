"""Run a frozen complete GPU diagnostic with full memory admission and emergency guards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def verify_registration(config):
    for entry in config['source_receipts']:
        if digest(ROOT/entry['path'])!=entry['sha256']:
            raise ValueError('Frozen full diagnostic source differs: '+entry['path'])
    settings=config['settings']
    if (settings['minimum_available_memory_bytes']<32*1024**3
        or settings['minimum_commit_headroom_bytes']<42*1024**3
        or settings['minimum_gpu_free_bytes']<14*1024**3
        or settings['emergency_commit_headroom_bytes']!=16*1024**3
        or settings['emergency_gpu_free_bytes']!=4*1024**3):
        raise ValueError('Full capacity admission gates changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    verify_registration(config)
    if args.probe_only:
        print(json.dumps(dict(torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
            original_full_model_batch_and_eight_workers_required=True,settings=config['settings'])));return
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'worker.stdout.log').exists():raise FileExistsError('Preserve previous full diagnostic attempt')
    api=guarded_api();owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Registered full diagnostic already controlled')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
               configuration_sha256=digest(ROOT/args.config),state='starting',active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with full_gpu_memory_slot(api,config,update):
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(config['command'],cwd=ROOT,stdout=out,stderr=err,
                    stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2'))
                handle=psutil.Process(child.pid)
                update(dict(state='running',active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,peaks=api['guarded_gpu_wait'](child,config['settings'],api,update)
        output=ROOT/config['expected_output']
        proof=json.loads(output.read_text(encoding='utf-8')) if output.exists() else None
        valid=bool(code==0 and proof and proof.get('passed') and proof.get('diagnostic_only')
                   and proof.get('queue_sha256')==config['expected_queue_fingerprint'])
        if valid:
            from scripts.flow_matching.run_spectral_table2_method_gated_queue_v1 import method_ready,read
            valid=method_ready(read(ROOT/config['queue']),'spectral_flow',
                               dict(method_proofs={'spectral_flow':config['expected_output']}))
        write(base/'resource_receipt.json',dict(exit_code=code,full_diagnostic_passed=valid,**peaks))
        update(dict(state='completed' if valid else 'failed_preserved',active_pid=None,
                    full_diagnostic_passed=valid,output_sha256=digest(output) if output.exists() else None))
        if not valid:raise RuntimeError('Complete original capacity diagnostic failed; preserve all evidence')
    finally:owner.close()


if __name__=='__main__':main()
