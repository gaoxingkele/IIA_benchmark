"""Resource-guarded complete native preparation, never benchmark training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    base = ROOT / config['output_directory']
    base.mkdir(parents=True, exist_ok=True)
    api = resource_api()
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Original GAN preparation controller is already active')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
                 active_pid=None, cases={}, config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 training_started=False, benchmark_performance=False, all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(base / 'status.json', state)

    try:
        for case in config['cases']:
            report = base / (case['id'] + '.json')
            if report.exists():
                state['cases'][case['id']] = 'existing_attempt_preserved'
                update({})
                continue
            update(dict(active_case=case['id']))
            with api['resource_slot'](base / 'preparation_family.lock', config['settings'], update):
                bootstrap = ("import runpy,sys;sys.path.insert(0," + repr(str(ROOT)) + ");"
                             "sys.argv=['audit_adopted_gan_native_v1','--config'," + repr(args.config)
                             + ",'--case'," + repr(case['id']) + "];"
                             "runpy.run_module('scripts.flow_matching.audit_adopted_gan_native_v1',run_name='__main__')")
                command = [config['python'], '-X', 'utf8', '-u', '-c', bootstrap]
                with (base / (case['id'] + '.stdout.log')).open('xb') as out, (base / (case['id'] + '.stderr.log')).open('xb') as err:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                 OMP_NUM_THREADS='2', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1'),
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_native_preparation', active_pid=process.pid,
                                active_create_time=psutil.Process(process.pid).create_time(), actual_command=command))
                    code, usage = capped_wait(process, config, api, update)
            passed = code == 0 and report.exists() and json.loads(report.read_text(encoding='utf-8'))['passed']
            write(base / (case['id'] + '.execution_receipt.json'),
                  dict(exit_code=code, complete_preparation_passed=passed, **usage))
            state['cases'][case['id']] = 'passed' if passed else 'failed_attempt_preserved'
            update(dict(active_pid=None, active_case=None, state='between_cases'))
        update(dict(state='preparation_completed' if all(s == 'passed' for s in state['cases'].values())
                    else 'preparation_completed_with_unresolved_cases'))
        print(json.dumps(state))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
