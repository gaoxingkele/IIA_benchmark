"""Run frozen full-result audit workers under a bounded read-only CPU lane."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, resource_api
from scripts.flow_matching.run_light_controller import digest
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def worker_command(config_path, config):
    return [config['python'], '-X', 'utf8', '-u', '-m',
            'scripts.flow_matching.audit_registered_grasp_results_v1', '--config', config_path, '--worker']


def verify_runtime(runtime):
    for entry in runtime['source_receipts']:
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen audit runtime source differs: '+entry['path'])
    ids = []
    for binding in runtime['audits']:
        path = ROOT/binding['config']
        if digest(path) != binding['config_sha256']:
            raise ValueError('Frozen original audit differs')
        config = read(path)
        if config['output_directory'] != binding['output_directory']:
            raise ValueError('Original audit output changed')
        for record in config['full_results']:
            if digest(ROOT/record['result_path']) != record['result_sha256']:
                raise ValueError('Complete registered native result differs')
            ids.append(record['id'])
    if len(set(ids)) != len(ids) or len(ids) != runtime['full_result_count']:
        raise ValueError('Duplicate original seed slots in audit lane')
    return ids


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    runtime=read(ROOT/args.runtime_config)
    ids=verify_runtime(runtime)
    if args.probe_only:
        print(json.dumps(dict(full_results=len(ids),torch_imported='torch' in sys.modules,
                             original_worker_and_scientific_inputs_unchanged=True)));return
    api=resource_api()
    base=ROOT/runtime['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Bounded audit lane already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',
               runtime_sha256=digest(ROOT/args.runtime_config),active_pid=None,audits={})
    def update(values):
        state.update(values);write(base/'status.json',state)
    try:
        for binding in runtime['audits']:
            config=read(ROOT/binding['config'])
            proof=ROOT/config['output_directory']/'independent_array_audit.json'
            if proof.exists():
                report=read(proof)
                if (not report['full_result_audit_passed'] or
                    {r['id'] for r in report['audited_full_results']} != {r['id'] for r in config['full_results']}):
                    raise ValueError('Existing audit proof does not cover full registered scope')
                state['audits'][binding['name']]='completed_preexisting';update({});continue
            if (ROOT/config['output_directory']).exists():
                raise FileExistsError('Preserve prior partial audit output')
            child_base=base/binding['name'];child_base.mkdir(exist_ok=True)
            command=worker_command(binding['config'],config)
            update(dict(active_audit=binding['name'],requested_command=command))
            with api['resource_slot'](ROOT/runtime['resource_lock'],runtime['settings'],update):
                with (child_base/'worker.stdout.log').open('xb') as out,(child_base/'worker.stderr.log').open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                        stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                            PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                    handle=psutil.Process(child.pid)
                    update(dict(state='auditing_complete_native_results',active_pid=child.pid,
                                active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                    code,usage=capped_wait(child,runtime,api,update)
            passed=bool(code==0 and proof.exists() and read(proof)['full_result_audit_passed'])
            write(child_base/'receipt.json',dict(exit_code=code,full_audit_passed=passed,**usage))
            state['audits'][binding['name']]='completed' if passed else 'failed_preserved'
            update(dict(active_pid=None,active_audit=None))
            if not passed: raise RuntimeError('Full audit failed; preserve all evidence and inputs')
        update(dict(state='completed',full_result_audit_passed=True))
        write(base/'receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
            full_result_audit_passed=True,full_result_count=len(ids),additional_training_seed_slots=0,
            original_worker_configs_and_result_outputs_unchanged=True))
    finally:
        owner.close()


if __name__=='__main__':main()
