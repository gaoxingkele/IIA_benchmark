"""Run all45 full native TimeGAN jobs with an isolated CPU family slot."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, verify_queue, verify_counts
from scripts.flow_matching.run_timegan_native_job_v1 import validate_evaluators
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait


def verify_result(queue, job):
    result = read(ROOT / job['output_directory'] / 'result.json')
    model = read(ROOT / queue['model_config'])
    verify_counts(result['optimizer_updates'], model)
    validate_evaluators(result['evaluators'])
    expected_shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    proof = read(ROOT / queue['construction_proof'])
    record = next(r for r in proof['records'] if r['dataset'] == job['dataset'] and r['module'] == job['module'])
    if (result['job'] != job or result['queue_sha256'] != sha(ROOT / queue['queue_path'])
            or not all(result[k] for k in ('full_original_source_main_completed', 'full_native_three_training_stages_completed',
                                          'full_native_evaluator_trainings_completed'))
            or result['full_generated_shape'] != expected_shape or result['full_original_data_shape'] != expected_shape
            or result['model_parameters'] != record['model_parameters']
            or len(result['checkpoints']) != 23 or len(result['artifacts']) != 22 or len(result['visualizations']) != 2
            or len(result['graph_seeds']) != 21 or result['independent_model_training_seed_slots'] != 1
            or result['author_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Complete original model/data/evaluator/output identities required')
    for item in result['artifacts'] + result['visualizations'] + [i for c in result['checkpoints'] for i in c['artifacts']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Full native result artifact changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    if args.queue != queue['queue_path']:
        raise ValueError('Frozen exact queue binding required')
    verify_queue(queue)
    if args.probe_only:
        print({'full_jobs': 45, 'tensorflow_imported': 'tensorflow' in __import__('sys').modules,
               'full_budget_preserved': True})
        return
    api = resource_api()
    base = ROOT / queue['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Native TimeGAN controller already active')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', jobs={}, active_pid=None,
                 queue_sha256=sha(ROOT / args.queue), all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
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
            with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
                # Embedded Python is isolated; explicitly add the project entry, not site-wide .pth changes.
                bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");"
                             "sys.argv=['run_timegan_native_job_v1','--queue'," + repr(args.queue) + ","
                             "'--job-id'," + repr(job['id']) + "];"
                             "runpy.run_module('scripts.flow_matching.run_timegan_native_job_v1',run_name='__main__')")
                command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / (job['id'] + '.stdout.log')).open('xb') as out, (logs / (job['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                                 OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads'])),
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_original_timegan', active_pid=child.pid,
                                active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                    # Use the established RSS/private/commit guard; rewrite only the display phase label.
                    def running(values):
                        update(dict(values, state='running_full_original_timegan'))
                    code, usage = capped_wait(child, queue, api, running)
            passed = False
            if code == 0 and (output / 'result.json').exists():
                verify_result(queue, job)
                passed = True
            write(base / 'receipts' / (job['id'] + '.json'), dict(exit_code=code, full_native_result_verified=passed, **usage))
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved'
            update(dict(state='between_full_jobs', active_pid=None, active_job=None))
            time.sleep(10)
        update(dict(state='completed' if all(v in ('completed', 'completed_preexisting') for v in state['jobs'].values())
                    else 'completed_with_unresolved_jobs'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
