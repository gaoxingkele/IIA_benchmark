"""Run all18 complete released Table4/5 commands with bounded data audit and GPU guards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_spectral_table45_original_v1 import ROOT, read, verify, verify_result, sha, write
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def command(queue, queue_path, job=None):
    args = [str(ROOT / queue['python']), '-X', 'utf8', '-u', '-m',
            'scripts.flow_matching.run_spectral_table45_original_v1', '--queue', queue_path]
    return args + (['--job-id', job['id']] if job else ['--audit-data'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args(); queue = read(ROOT / args.queue); verify(queue)
    if args.probe_only:
        print(json.dumps(dict(full_original_jobs=len(queue['jobs']), torch_imported='torch' in sys.modules,
            numpy_imported='numpy' in sys.modules, original_epochs=600, original_batch_size=64)))
        return
    api = guarded_api(); base = ROOT / queue['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full Table4/5 controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), queue_sha256=sha(ROOT / args.queue),
                 state='starting', active_pid=None, active_job=None, jobs={})
    def update(values):
        state.update(values); write(base / 'status.json', state)
    def launch(cmd, logname, cpu=False):
        logs = base / 'logs'; logs.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2')
        env['CUDA_VISIBLE_DEVICES'] = '-1' if cpu else str(queue['settings']['gpu_index'])
        with (logs / (logname+'.stdout.log')).open('xb') as out, (logs / (logname+'.stderr.log')).open('xb') as err:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            handle = psutil.Process(child.pid)
            update(dict(state='auditing_full_original_data' if cpu else 'running_full_original_job',
                        active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
            if cpu:
                return capped_wait(child, {'settings':queue['data_audit_settings']}, api, update)
            return api['guarded_gpu_wait'](child, queue['settings'], api, update)
    try:
        proof_path = ROOT / queue['data_audit_output']
        if not proof_path.exists():
            with api['resource_slot'](ROOT/queue['data_audit_lock'], queue['data_audit_settings'], update):
                code, peaks = launch(command(queue, args.queue), 'full_data_audit', cpu=True)
            passed = code == 0 and proof_path.exists() and read(proof_path)['passed']
            write(base/'data_audit_resource_receipt.json', dict(exit_code=code, passed=passed, **peaks))
            update(dict(active_pid=None))
            if not passed:
                update(dict(state='failed_data_audit_preserved'))
                raise RuntimeError('Full original data audit failed; keep raw/workspace/logs')
        proof = read(proof_path)
        if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256'] != sha(ROOT/args.queue):
            raise ValueError('Frozen full original data proof differs')
        update(dict(state='full_data_audit_completed', data_audit_sha256=sha(proof_path), active_pid=None))
        if args.audit_only:
            return
        for job in queue['jobs']:
            result_path = ROOT/job['output_directory']/'result.json'
            if result_path.exists():
                verify_result(queue, job); state['jobs'][job['id']] = 'completed_preexisting'; continue
            if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                state['jobs'][job['id']] = 'failed_or_partial_preserved'; update({}); continue
            update(dict(active_job=job['id']))
            with full_gpu_memory_slot(api, queue, update):
                code, peaks = launch(command(queue, args.queue, job), job['id'])
            valid = False
            if code == 0 and result_path.exists():
                verify_result(queue, job); valid = True
            state['jobs'][job['id']] = 'completed' if valid else 'failed_or_partial_preserved'
            write(base/'receipts'/(job['id']+'.json'), dict(exit_code=code, full_result_validated=valid, **peaks))
            update(dict(state='between_original_jobs', active_pid=None, active_job=None))
            time.sleep(queue['settings']['controller_yield_seconds'])
        complete = all(v in ('completed', 'completed_preexisting') for v in state['jobs'].values())
        update(dict(state='completed' if complete else 'completed_with_unresolved_jobs', all18_original_jobs_completed=complete))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
