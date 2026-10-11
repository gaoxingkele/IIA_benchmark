"""Execute frozen full released-data C-RNN-GAN variants without shortened budgets."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def verify_registration(queue):
    for receipt in queue['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen complete native training registration changed: ' + receipt['path'])
    expected = {(variant, seed) for variant in queue['variants'] for seed in range(1103, 1108)}
    actual = [(j['variant'], j['seed']) for j in queue['jobs']]
    if (len(queue['variants']) != 7 or len(actual) != 35 or set(actual) != expected
            or len({j['id'] for j in queue['jobs']}) != 35
            or len({j['output_directory'] for j in queue['jobs']}) != 35):
        raise ValueError('All seven registered full models/controls and five canonical seeds required')
    flags = queue['flags']
    if (flags['max_epoch'] != 500 or flags['songlength'] != 100 or flags['pretraining_epochs'] != 6
            or flags['hidden_size_g'] != 350 or flags['hidden_size_d'] != 350
            or flags['num_layers_g'] != 2 or flags['num_layers_d'] != 2 or flags['batch_size'] != 20
            or flags['unidirectional_d'] or flags['exit_after'] != 0
            or flags['works_per_composer'] is not None or flags['synthetic_chords']):
        raise ValueError('Complete original source budget, paper model geometry and all released files required')
    for job in queue['jobs']:
        variant = queue['variants'][job['variant']]
        if job['tones_per_cell'] != variant['tones_per_cell'] or job['variant_flags'] != variant['flags']:
            raise ValueError('Frozen full variant parameters changed')
        model = read(ROOT / variant['model_config'])
        if model['parameters'] != dict(flags, **job['variant_flags'], tones_per_cell=job['tones_per_cell']):
            raise ValueError('The model config must be the source of every training flag')
    if queue['parsed_released_files'] != 3014 or queue['full_note_events'] != 10323257:
        raise ValueError('Complete released MIDI corpus required')


def verify_result(queue, job):
    output = ROOT / job['output_directory']
    result = read(output / 'result.json')
    if (result['job'] != job or not result['full500_epochs_completed'] or not result['full_training_completed']
            or len(result['epochs']) != 500 or result['author_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Complete native full training required; do not promote alignment claims')
    loader = result['full_data_audit']
    if loader['full_parsed_compositions'] != 3014 or loader['full_note_events'] != 10323257:
        raise ValueError('All originally parsed released data required')
    pretraining_epochs = job['variant_flags'].get('pretraining_epochs', queue['flags']['pretraining_epochs'])
    batches = loader['split_counts']['train'] // queue['flags']['batch_size']
    if result['actual_optimizer_updates']['pretraining'] != pretraining_epochs * batches:
        raise ValueError('Every original full pretraining minibatch required')
    for index, epoch in enumerate(result['epochs']):
        if epoch['epoch'] != index or epoch['full_training_batches'] != batches:
            raise ValueError('Every full epoch and grouped composition traversal required')
        for receipt in epoch['checkpoint_artifacts']:
            if sha(receipt['path']) != receipt['sha256']:
                raise ValueError('Full native checkpoint/RNG artifacts changed')
    if result['full_test_audit']['full_batches'] != loader['split_counts']['test'] // queue['flags']['batch_size']:
        raise ValueError('Complete original final test required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    verify_registration(queue)
    if args.probe_only:
        print(dict(full_native_jobs=35, epochs_each=500, tensorflow_imported='tensorflow' in sys.modules))
        return
    if args.worker:
        from scripts.flow_matching.crnngan_native_training_runtime_v1 import run
        job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
        run(queue, job)
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
        raise RuntimeError('Complete native C-RNN-GAN controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None,
                 jobs={}, queue_sha256=sha(ROOT / args.queue), all_original_experiments_complete=False)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(base / 'status.json', state)

    try:
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
                command = [queue['python'], '-X', 'utf8', '-u', '-c',
                    "import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");sys.argv=['crnngan_queue','--queue',"
                    + repr(args.queue) + ",'--worker','--job-id'," + repr(job['id'])
                    + "];runpy.run_module('scripts.flow_matching.run_crnngan_native_original_queue_v2',run_name='__main__')"]
                logs = base / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / (job['id'] + '.stdout.log')).open('xb') as out, (logs / (job['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                                 OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    update(dict(state='running_full_native_crnngan', active_job=job['id'], active_pid=child.pid,
                        active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))

                    def running(values):
                        update(dict(values, state='running_full_native_crnngan'))

                    code, usage = capped_wait(child, queue, api, running)
            passed = False
            if code == 0:
                verify_result(queue, job)
                passed = True
            write(base / 'receipts' / (job['id'] + '.json'), dict(exit_code=code, full_native_result_verified=passed, **usage))
            state['jobs'][job['id']] = 'completed' if passed else 'failed_or_partial_preserved'
            update(dict(state='between_jobs', active_pid=None, active_job=None))
            if not passed:
                update(dict(state='stopped_for_failed_job_diagnosis'))
                return
            time.sleep(10)
        update(dict(state='completed' if all(v in ('completed', 'completed_existing') for v in state['jobs'].values())
                    else 'completed_with_unresolved_jobs'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
