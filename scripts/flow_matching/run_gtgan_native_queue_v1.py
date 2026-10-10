"""Schedule all four full native stock commands behind the shared GPU mutex."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_gtgan_native_job_v1 import ROOT, sha, full_budget


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def verify_result(queue, job):
    result = json.loads((ROOT / job['output_directory'] / 'result.json').read_text(encoding='utf-8'))
    expected = full_budget(result['case'])
    if (result['job'] != job or result['queue_sha256'] != sha(ROOT / queue['queue_path']) or
            not result['full_native_command_completed'] or result['author_equivalence_certified'] or
            result['expected_budget'] != expected or len(result['evaluations']) != expected['full_periodic_evaluations']):
        raise ValueError('Complete frozen native command result required')
    counts = result['optimizer_updates']
    if counts['embedding'] != expected['embedding_optimizer_updates'] or counts['generator'] != expected['generator_optimizer_updates']:
        raise ValueError('Full native training optimizer counts missing')
    for index, row in enumerate(result['evaluations'], 1):
        if (row['native_joint_step'] != 500 * index or row['full_generation_windows'] != 3662 or
                len(row['predictive_repeats']) != 10 or len(row['discriminative_repeats']) != 10 or
                sha(row['artifact_path']) != row['artifact_sha256']):
            raise ValueError('Complete native full-dataset evaluator evidence differs')
    if len(result['checkpoints']) != 4 * expected['full_periodic_evaluations']:
        raise ValueError('All native network checkpoints required')
    for item in result['checkpoints']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Native trained checkpoint changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    args = parser.parse_args()
    queue = json.loads((ROOT / args.queue).read_text(encoding='utf-8'))
    if queue['queue_path'] != args.queue:
        raise ValueError('Immutable queue binding differs')
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen native queue source changed: ' + item['path'])
    cases = json.loads((ROOT / queue['model_config']).read_text(encoding='utf-8'))['cases']
    if len(queue['jobs']) != 4 or {j['case_id'] for j in queue['jobs']} != {c['id'] for c in cases}:
        raise ValueError('All four original stock missing-rate commands required')
    for case in cases:
        full_budget(case)
    api = guarded_api()
    base = ROOT / queue['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Original GT-GAN controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                 queue_sha256=sha(ROOT / args.queue), state='starting_full_native_commands',
                 jobs={}, active_pid=None, active_job=None)

    def update(value):
        state.update(value)
        write(base / 'status.json', state)

    try:
        for job in queue['jobs']:
            output = ROOT / job['output_directory']
            if (output / 'result.json').exists():
                verify_result(queue, job)
                state['jobs'][job['id']] = 'completed_preexisting'
                continue
            if output.exists():
                state['jobs'][job['id']] = 'failed_or_partial_preserved'
                update({})
                continue
            update(dict(active_job=job['id']))
            with full_gpu_memory_slot(api, queue, update):
                command = [str(ROOT / queue['python']), '-X', 'utf8', '-u', '-m',
                           'scripts.flow_matching.run_gtgan_native_job_v1', '--queue', args.queue,
                           '--job-id', job['id']]
                env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                           PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', MPLBACKEND='Agg',
                           OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads']),
                           TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS=str(queue['cpu_threads']))
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / (job['id'] + '.stdout.log')).open('xb') as out, (logs / (job['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_original_native_command', active_pid=child.pid,
                                active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                    code, usage = api['guarded_gpu_wait'](child, queue['settings'], api, update)
            passed = False
            if code == 0 and (output / 'result.json').exists():
                verify_result(queue, job)
                passed = True
            write(base / 'receipts' / (job['id'] + '.json'), dict(exit_code=code,
                  full_native_result_verified=passed, **usage))
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved'
            update(dict(state='between_full_native_commands', active_pid=None, active_job=None))
            time.sleep(queue['settings']['controller_yield_seconds'])
        update(dict(state='completed' if len(state['jobs']) == 4 and
                    all(v in ('completed', 'completed_preexisting') for v in state['jobs'].values())
                    else 'completed_with_unresolved_jobs'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
