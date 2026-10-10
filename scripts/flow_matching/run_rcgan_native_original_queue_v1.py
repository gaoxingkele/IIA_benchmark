"""Run all five full500-epoch native RCGAN MNIST jobs after evaluator proof."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def verify_registration(queue):
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full original MNIST registration changed')
    if len(queue['jobs']) != 5 or {job['seed'] for job in queue['jobs']} != set(range(1103, 1108)):
        raise ValueError('All five original training seed obligations required')
    if (len({job['id'] for job in queue['jobs']}) != 5 or len({job['output_directory'] for job in queue['jobs']}) != 5
            or any(job['id'] != 'rcgan_mnistfull__seed' + str(job['seed']) for job in queue['jobs'])):
        raise ValueError('Distinct canonical seed/output identities required')
    if (queue['epochs'] != 500 or queue['D_updates_per_epoch'] != 1284 or queue['G_updates_per_epoch'] != 1284
            or queue['sigma_maximum_updates_per_evaluation'] != 2000 or queue['full_evaluations_per_job'] != 500):
        raise ValueError('Complete original GAN and evaluator budgets required')


def verify_result(queue, job):
    result = read(ROOT / job['output_directory'] / 'result.json')
    if (result['job'] != job or not result['full_original500_epoch_loop_completed']
            or not result['full_training_completed'] or not result['full_original_evaluators_completed']
            or len(result['epochs']) != 500 or len(result['evaluators']) != 500 or len(result['checkpoints']) != 500
            or result['optimizer_updates']['D'] != 642000 or result['optimizer_updates']['G'] != 642000
            or result['full_original_split_shapes'] != queue['full_split_shapes']
            or result['independent_training_seed_slots'] != 1 or result['author_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Complete full original model/data/epochs/evaluator budgets required')
    for index, evaluation in enumerate(result['evaluators']):
        if (evaluation['epoch'] != index or evaluation['complete_generated_points'] != 4984
                or evaluation['full_fit_points'] != 996 or evaluation['full_test_points'] != 3988
                or not 1 <= evaluation['sigma_optimizer_updates'] <= 2000
                or (evaluation['sigma_optimizer_updates'] < 2000 and evaluation['stopping_delta'] > .001)):
            raise ValueError('Every original full native sigma stopping budget required')
        if sha(evaluation['array_path']) != evaluation['array_sha256']:
            raise ValueError('Full original evaluator arrays changed')
    for index, checkpoint in enumerate(result['checkpoints']):
        if checkpoint['epoch'] != index or any(checkpoint['optimizer_updates'][k] != (index + 1) * 1284 for k in ('D', 'G')):
            raise ValueError('Every native full checkpoint/update ledger required')
        for artifact in checkpoint['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Full original checkpoint or RNG state changed')
    loader = result['full_native_loader_audit']
    if (not loader['grouped_disjoint_image_indices'] or not loader['every_original_split_pixel_and_label_bitwise_verified']
            or loader['full_original_training_images'] != 60000
            or sha(loader['full_split_indices_path']) != loader['full_split_indices_sha256']):
        raise ValueError('Complete original data audit and grouped image split required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--proof-binding')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    verify_registration(queue)
    if args.probe_only:
        print(dict(full_native_jobs=5, epochs_each=500, tensorflow_imported='tensorflow' in sys.modules,
                   full_original_evaluator_proof_required_before_dispatch=True))
        return
    if args.worker:
        from scripts.flow_matching.rcgan_native_training_runtime_v1 import run
        binding = read(ROOT / args.proof_binding)
        if binding['queue_sha256'] != sha(ROOT / args.queue) or binding['proof_sha256'] != sha(ROOT / queue['native_evaluator_proof']):
            raise ValueError('Dispatch must retain the frozen full native evaluator proof')
        execution = read(ROOT / queue['native_evaluator_execution_receipt'])
        if execution['exit_code'] != 0 or not execution['complete_preparation_passed']:
            raise ValueError('Successful complete native evaluator execution required')
        bound_queue = dict(queue, native_evaluator_proof_sha256=binding['proof_sha256'])
        job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
        run(bound_queue, job)
        verify_result(queue, job)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    base = ROOT / queue['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Native full MNIST controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None,
                 jobs={}, queue_sha256=sha(ROOT / args.queue), all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(base / 'status.json', state)

    try:
        while not ((ROOT / queue['native_evaluator_proof']).exists()
                   and (ROOT / queue['native_evaluator_execution_receipt']).exists()):
            launch = read(ROOT / queue['native_evaluator_launch'])
            try:
                process = psutil.Process(launch['pid'])
                identity = process.create_time() == launch['create_time'] and process.cmdline() == launch['actual_command']
            except psutil.NoSuchProcess:
                identity = False
            if not identity:
                update(dict(state='pending_missing_or_failed_native_evaluator_audit', evaluator_controller_verified_live=False))
                raise RuntimeError('Full native evaluator missing/stopped before successful proof; preserve every attempt')
            update(dict(state='waiting_for_registered_full_native_evaluator_audit', evaluator_controller_verified_live=True,
                        evaluator_pid=launch['pid'], evaluator_create_time=launch['create_time']))
            time.sleep(10)
        execution = read(ROOT / queue['native_evaluator_execution_receipt'])
        proof = read(ROOT / queue['native_evaluator_proof'])
        if execution['exit_code'] != 0 or not execution['complete_preparation_passed'] or not proof['passed']:
            update(dict(state='pending_failed_native_evaluator_audit'))
            raise ValueError('Full native evaluator must actually pass before model training')
        binding_path = base / 'full_native_evaluator_proof_binding.json'
        write(binding_path, dict(queue_sha256=sha(ROOT / args.queue), proof_sha256=sha(ROOT / queue['native_evaluator_proof']),
            native_execution_receipt_sha256=sha(ROOT / queue['native_evaluator_execution_receipt'])))
        for job in queue['jobs']:
            output = ROOT / job['output_directory']
            if (output / 'result.json').exists():
                verify_result(queue, job)
                state['jobs'][job['id']] = 'completed_existing'
                continue
            if output.exists():
                state['jobs'][job['id']] = 'failed_or_partial_preserved'
                update({})
                continue
            with api['resource_slot'](ROOT / queue['resource_lock'], queue['settings'], update):
                bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");"
                    "sys.argv=['run_rcgan_native_original_queue_v1','--queue'," + repr(args.queue)
                    + ",'--worker','--job-id'," + repr(job['id']) + ",'--proof-binding'," + repr(str(binding_path)) + "];"
                    "runpy.run_module('scripts.flow_matching.run_rcgan_native_original_queue_v1',run_name='__main__')")
                command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / (job['id'] + '.stdout.log')).open('xb') as out, (logs / (job['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                                 OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_native_rcgan_mnist500', active_job=job['id'], active_pid=child.pid,
                        active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))

                    def running(values):
                        update(dict(values, state='running_full_native_rcgan_mnist500'))

                    code, usage = capped_wait(child, queue, api, running)
            passed = False
            if code == 0:
                verify_result(queue, job)
                passed = True
            write(base / 'receipts' / (job['id'] + '.json'), dict(exit_code=code, full_native_result_verified=passed, **usage))
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved'
            update(dict(state='between_jobs', active_pid=None, active_job=None))
            time.sleep(10)
        update(dict(state='completed' if all(s in ('completed', 'completed_existing') for s in state['jobs'].values())
                    else 'completed_with_unresolved_jobs'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
