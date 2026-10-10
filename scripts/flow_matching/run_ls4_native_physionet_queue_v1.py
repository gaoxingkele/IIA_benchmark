"""Queue both full native LS4 patient tasks with unchanged full GPU safeguards."""
import argparse
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.run_ls4_native_physionet_v1 import budget, periodic_epochs


def verify_result(queue, job):
    output = ROOT / job['output_directory']
    result = read(output / 'result.json')
    config = read(ROOT / job['model_config'])
    expected = budget(config)
    if (result['job'] != job or result['queue_sha256'] != sha(ROOT / queue['queue_path'])
            or result['config_sha256'] != sha(ROOT / job['model_config'])
            or result['epochs_completed'] != 500 or result['optimizer_updates'] != expected['optimizer_updates']
            or [r['epoch'] for r in result['evaluations']] != periodic_epochs(500, 10) + [-1]
            or result['author_equivalence_certified']):
        raise ValueError('Complete frozen native PhysioNet scope required')
    for row in result['evaluations']:
        if row['full_test_patients'] != 1600 or row['full_test_batches'] != expected['test_batches']:
            raise ValueError('Full original test coverage missing')
    for artifact in result['artifacts']:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Native trained-model/evaluator artifact changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned patient queue source changed')
    api = guarded_api()
    base = ROOT / queue['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Native patient controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
        queue_sha256=sha(ROOT / args.queue), active_pid=None, active_job=None, jobs={})
    def update(values):
        state.update(values); write(base / 'status.json', state)
    def launch(name, profile=None, job=None, cpu=False):
        command = [str(ROOT / queue['python']), '-X', 'utf8', '-u', '-m',
            'scripts.flow_matching.run_ls4_native_physionet_v1', '--queue', args.queue]
        command += ['--job-id', job['id']] if job else ['--integration-check' if cpu else '--gpu-preflight']
        logs = base / 'logs'; logs.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads']),
            WANDB_MODE='disabled', MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='-1' if cpu else '0',
            LS4_PHYSIONET_PROFILE=profile or job['profile'])
        with (logs / (name+'.stdout.log')).open('xb') as out, (logs / (name+'.stderr.log')).open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            update(dict(state='running_original_model_interface' if cpu else 'running_full_original_gpu_task',
                active_pid=child.pid, active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
            return capped_wait(child, dict(settings=queue['integration_settings']), api, update) if cpu else api['guarded_gpu_wait'](child, queue['settings'], api, update)
    try:
        for name, cpu in [('integration', True), ('gpu_preflight', False)]:
            for profile in queue['profiles']:
                proof_path = ROOT / queue[name+'_output'] / (profile+'.json')
                label = name+'__'+profile
                if proof_path.exists():
                    proof = read(proof_path)
                    if not proof['passed'] or not proof['diagnostic_only'] or proof['queue_sha256'] != sha(ROOT / args.queue):
                        raise ValueError('Immutable full patient diagnostic differs')
                    continue
                if (base/'logs'/(label+'.stdout.log')).exists():
                    raise FileExistsError('Preserve failed/partial patient diagnostic')
                slot = api['resource_slot'](ROOT/queue['integration_lock'], queue['integration_settings'], update) if cpu else full_gpu_memory_slot(api, queue, update)
                with slot:
                    code, usage = launch(label, profile=profile, cpu=cpu)
                passed = code == 0 and proof_path.exists() and read(proof_path)['passed']
                write(base/'receipts'/(label+'.json'), dict(exit_code=code, passed=passed, **usage))
                update(dict(active_pid=None))
                if not passed:
                    raise RuntimeError('Original patient interface failed; partial artifacts preserved')
        for job in queue['jobs']:
            if (ROOT/job['output_directory']/'result.json').exists():
                verify_result(queue, job); state['jobs'][job['id']] = 'completed_preexisting'; continue
            if (ROOT/job['output_directory']).exists() or Path(job['artifact_directory']).exists():
                state['jobs'][job['id']] = 'failed_or_partial_preserved'; update({}); continue
            update(dict(active_job=job['id']))
            with full_gpu_memory_slot(api, queue, update):
                code, usage = launch(job['id'], job=job)
            passed = False
            if code == 0 and (ROOT/job['output_directory']/'result.json').exists():
                verify_result(queue, job); passed = True
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved'
            write(base/'receipts'/(job['id']+'.json'), dict(exit_code=code, full_result_validated=passed, **usage))
            update(dict(active_pid=None, active_job=None, state='between_original_patient_jobs'))
            time.sleep(queue['settings']['controller_yield_seconds'])
        update(dict(state='completed' if len(state['jobs']) == 10 and all(v in ('completed','completed_preexisting') for v in state['jobs'].values()) else 'completed_with_unresolved_jobs'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
