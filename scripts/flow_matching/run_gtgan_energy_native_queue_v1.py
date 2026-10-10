"""Schedule all four exact original Energy commands behind the shared GPU mutex."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.run_spectral_original_queue import guarded_api
from scripts.flow_matching.run_gtgan_energy_native_job_v1 import ROOT, sha, full_budget


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def verify_result(queue, job):
    result = json.loads((ROOT / job['output_directory'] / 'result.json').read_text(encoding='utf-8'))
    model = json.loads((ROOT / queue['model_config']).read_text(encoding='utf-8'))
    registered = next(c for c in model['cases'] + model['release_evaluation_cases'] if c['id'] == job['case_id'])
    if result['case'] != registered:
        raise ValueError('The actual result must bind to the exact registered Energy case')
    expected = full_budget(result['case'])
    if (result['job'] != job or result['queue_sha256'] != sha(ROOT / queue['queue_path']) or
            result['construction_proof_sha256'] != sha(ROOT / queue['construction_proof']) or
            not result['full_native_command_completed'] or result['author_equivalence_certified'] or
            result['expected_budget'] != expected or len(result['evaluations']) != expected['full_periodic_evaluations']):
        raise ValueError('Complete frozen native command result required')
    counts = result['optimizer_updates']
    if counts['embedding'] != expected['embedding_optimizer_updates'] or counts['generator'] != expected['generator_optimizer_updates']:
        raise ValueError('Full native training optimizer counts missing')
    if not 0 <= counts['discriminator'] <= expected['maximum_discriminator_updates']:
        raise ValueError('The original conditional discriminator count differs')
    for index, row in enumerate(result['evaluations'], 1):
        step = 0 if expected['native_release_checkpoint_evaluation'] else 500 * index
        if (row['native_joint_step'] != step or row['full_generation_windows'] != 19712 or
                len(row['predictive_repeats']) != 10 or len(row['discriminative_repeats']) != 10 or
                sha(row['artifact_path']) != row['artifact_sha256']):
            raise ValueError('Complete native full-dataset evaluator evidence differs')
        sampler = row['sampler_input']
        if sampler['full_windows'] != expected['full_generation_windows'] or sha(sampler['path']) != sampler['sha256']:
            raise ValueError('Full actual native latent noise, times and RNG states required')
        evaluators = row['evaluator_training_receipts']
        if len(evaluators) != 20 or [e['role'] for e in evaluators] != ['discriminative'] * 10 + ['predictive'] * 10:
            raise ValueError('All ten original evaluator trainings per metric are required')
        for evaluator in evaluators:
            steps = expected[evaluator['role'] + '_evaluator_steps']
            if (evaluator['optimizer_updates'] != steps or not evaluator['full_saved_prediction_score_check_passed']
                    or sha(evaluator['array_path']) != evaluator['array_sha256']
                    or len(evaluator['checkpoint_files']) != 3):
                raise ValueError('Original full post-hoc evaluator state and test outputs required')
            for item in evaluator['checkpoint_files']:
                if sha(item['path']) != item['sha256']:
                    raise ValueError('Actual trained evaluator checkpoint changed')
    expected_checkpoints = 0 if expected['native_release_checkpoint_evaluation'] else 4 * expected['full_periodic_evaluations']
    if len(result['checkpoints']) != expected_checkpoints:
        raise ValueError('All native network checkpoints required')
    for item in result['checkpoints']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Native trained checkpoint changed')
    if result['training_seeds'] != (0 if expected['native_release_checkpoint_evaluation'] else 1):
        raise ValueError('Release evaluation must not be counted as generator training')
    if result['full_native_generator_training_completed'] is not (not expected['native_release_checkpoint_evaluation']):
        raise ValueError('Completed command and completed generator training must remain distinct')
    if expected['native_release_checkpoint_evaluation']:
        if len(result['released_checkpoints']) != 2:
            raise ValueError('Both actual published Energy checkpoint inputs required')
        for item in result['released_checkpoints']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Released checkpoint input changed')
    return result


def construction_ready(queue):
    path = ROOT / queue['construction_proof']
    receipt_path = ROOT / queue['construction_audit_state_root'] / 'receipt.json'
    if not path.exists() or not receipt_path.exists():
        return False
    proof = json.loads(path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt['exit_code'] != 0 or not receipt['full_native_energy_audit_passed']
            or not proof['passed'] or not proof['full_raw_data_processing_verified']
            or proof['full_original_constructor_cases'] != 2 or proof['performance_result']
            or proof['author_equivalence_certified'] or not proof['raw_csv_unchanged']
            or proof['cuda_context_initialized'] or proof['config_sha256'] != queue['construction_audit_config_sha256']):
        raise ValueError('Complete native Energy proof and successful actual worker exit required')
    if {c['missing_rate'] for c in proof['cases']} != {0.0, 0.5}:
        raise ValueError('Both original missing-rate profiles required')
    for case in proof['cases']:
        if (case['complete_windows'] != 19712 or case['variables'] != 28 or case['raw_rows'] != 19735
                or not case['full_original_cubic_coefficients_finite'] or not case['all_observed_values_exact']
                or len(case['released_weight_checks']) != 2 or sha(case['artifact_path']) != case['artifact_sha256']):
            raise ValueError('Complete native arrays and matching released-weight checks required')
    return True


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
    model = json.loads((ROOT / queue['model_config']).read_text(encoding='utf-8'))
    cases = model['cases'] + model['release_evaluation_cases']
    if len(queue['jobs']) != 4 or {j['case_id'] for j in queue['jobs']} != {c['id'] for c in cases}:
        raise ValueError('All four exact original README Energy commands required')
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
        while not construction_ready(queue):
            expected = queue['construction_audit_parent']
            try:
                parent = psutil.Process(expected['pid'])
                live = parent.create_time() == expected['create_time'] and queue['construction_audit_config'] in parent.cmdline()
            except psutil.NoSuchProcess:
                live = False
            if not live:
                if construction_ready(queue):
                    break
                raise RuntimeError('The exact full audit stopped without successful complete evidence; do not launch or restart')
            update(dict(state='waiting_full_native_energy_data_and_model_proof',
                construction_audit_parent_verified_live=True, generator_training_started=False))
            time.sleep(5)
        update(dict(full_native_data_model_proof_verified=True,
            construction_proof_sha256=sha(ROOT / queue['construction_proof'])))
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
                           'scripts.flow_matching.run_gtgan_energy_native_job_v1', '--queue', args.queue,
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
