"""Verify complete original Sine15000 joint state before any continuation.

TimeGAN, Yoon et al., NeurIPS2019. All284 model/Adam variables and both RNG
states are restored in the frozen TF1.15 environment, with zero training.
This proof cannot certify the remaining35000 joint iterations or all20 metrics.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.timegan_stage_resume_v2 import binding, restore_all, numpy_state
from scripts.flow_matching.timegan_joint_checkpoint_observer_v1 import expected_joint_counts, equal_numpy_state


def tuple_state(value):
    return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen original saved-joint binding changed: ' + pin['path'])
    previous = read(ROOT / config['previous_recovery_configuration'])
    queue, model, job, stages = binding(previous)
    expected = dict(queue['settings'], minimum_commit_headroom_bytes=40 * 1024**3)
    receipt = read(ROOT / config['joint_checkpoint_receipt'])
    terminal = read(ROOT / config['terminal_receipt'])
    if (config['canonical_job'] != job or config['settings'] != expected
            or config['resource_lock'] != queue['resource_lock'] or config['python'] != queue['python']
            or config['cpu_threads'] != queue['cpu_threads']
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
            or receipt['canonical_job_id'] != job['id'] or receipt['completed_joint_iterations'] != 15000
            or receipt['next_original_joint_iteration'] != 15000 or receipt['required_joint_iterations'] != 50000
            or not expected_joint_counts(receipt['optimizer_updates'], 15000)
            or receipt['native_full_variable_count'] != previous['checkpoint_full_variable_count']
            or receipt['every_saved_native_model_optimizer_value_bitwise_equal'] is not True
            or receipt['model_optimizer_numpy_python_rng_and_counters_unchanged_by_observer'] is not True
            or receipt['untracked_training_or_generation_TF_random_operations']
            or terminal['exit_code'] == 0 or terminal['resource_guard_aborted'] is not True):
        raise ValueError('Actual full15000 state and stronger admission only required')
    ledger = read(ROOT / config['completed_joint_ledger'])
    if [r['completed_joint_iterations'] for r in ledger] != list(range(1000, 15001, 1000)) or ledger[-1] != receipt:
        raise ValueError('Complete monotonic original joint-state ledger required')
    for record in ledger:
        for artifact in record['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Saved full original joint checkpoint/RNG changed')
    return queue, model, job, receipt


def worker(config):
    queue, model, job, receipt = verify(config)
    environment = read(ROOT / queue['environment_lock'])
    for name, key in [('python.exe', 'python_executable_sha256'), ('python37.dll', 'python37_dll_sha256'),
                      ('python37._pth', 'python_pth_sha256')]:
        if sha(Path(environment['runtime_root']) / name) != environment[key]:
            raise ValueError('Frozen native interpreter differs')
    import numpy as np
    import tensorflow as tf
    import pkg_resources
    package_root = Path(environment['runtime_root']) / 'Lib/site-packages'
    packages = {p.key: p.version for p in pkg_resources.working_set if Path(p.location).resolve() == package_root.resolve()}
    if (Path(sys.executable).resolve() != Path(queue['python']).resolve() or sys.version != environment['python']
            or packages != environment['packages'] or tf.__version__ != '1.15.0' or np.__version__ != '1.18.5'):
        raise ValueError('Exact original numerical environment required')
    from scripts.flow_matching.timegan_native_common_v1 import verify_source, constructor_prefix
    from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import independent_full_loader
    source, before = verify_source(model)
    os.chdir(str(source)); sys.path.insert(0, str(source))
    import timegan as author
    import data_loading as loader
    real, data_audit = independent_full_loader(job, model, np, loader)
    tf.reset_default_graph(); tf.set_random_seed(job['seed'])
    nodes = constructor_prefix(author, 'timegan')(real,
        dict(module=job['module'], hidden_dim=24, num_layer=3, iterations=50000, batch_size=128))
    directory = Path(config['joint_checkpoint_receipt']).parent
    if not directory.is_absolute():
        directory = ROOT / directory
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0},
            intra_op_parallelism_threads=queue['cpu_threads'], inter_op_parallelism_threads=1)) as session:
        proof = restore_all(session, nodes, directory / 'model', directory / 'numpy_random_state.npz')
        expected_py = tuple_state(read(directory / 'python_random_state.json'))
        random.setstate(expected_py)
        if (random.getstate() != expected_py
                or not equal_numpy_state(np.random.get_state(), numpy_state(directory / 'numpy_random_state.npz'), np)
                or proof['restored_native_variables'] != receipt['native_full_variable_count']
                or int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables())) != receipt['model_parameters']):
            raise ValueError('Every original model/Adam/RNG value must restore exactly')
    # The constructor creates graph operations but executes no optimizer.
    target = ROOT / config['output_directory']; target.mkdir(parents=True, exist_ok=False)
    write(target / 'saved_joint15000_native_restore.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job_id=job['id'],
        completed_joint_iterations=15000, remaining_original_joint_iterations=35000,
        required_original_joint_iterations=50000, saved_optimizer_updates=receipt['optimizer_updates'],
        full_native_parameters=receipt['model_parameters'], full_original_data_shape=list(real.shape),
        data_audit=data_audit, native_restore=proof, native_NumPy_and_Python_RNG_bitwise_restored=True,
        new_optimizer_updates_executed=0, additional_training_seed_slots=0,
        full_joint_continuation_executed=False, full20_evaluator_trainings_completed=False,
        independent_full45_replay_completed=False, all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config)
    queue, model, job, saved = verify(config)
    if args.probe_only:
        print(json.dumps(dict(registration_verified=True, original_jobs=45,
            completed_saved_joint_iterations=15000, remaining_joint_iterations=35000,
            tensorflow_imported='tensorflow' in sys.modules, training_executed=False)))
        return
    if args.worker:
        worker(config); return
    import psutil
    from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('This native saved-state audit already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=sha(ROOT / args.config), state='starting', all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat());write(base / 'status.json', state)

    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            if sha(ROOT / args.config) != state['configuration_sha256']:
                raise ValueError('Live saved-state audit configuration changed')
            verify(config)
            argv = ['native_saved_joint_audit', '--config', args.config, '--worker']
            bootstrap = ('import sys,runpy;sys.path.insert(0,' + repr(str(ROOT)) + ');sys.argv='
                + repr(argv) + ";runpy.run_module('scripts.flow_matching.audit_timegan_sine_saved_joint15000_v1',run_name='__main__')")
            command = [config['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(active_pid=child.pid, active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        target = ROOT / config['output_directory'] / 'saved_joint15000_native_restore.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and target.exists() and read(target)['passed'] is True
        write(base / 'receipt.json', dict(exit_code=code, full_native_joint15000_restore_passed=passed,
            full_original_training_completed=False, **usage))
        update(dict(active_pid=None, state='native_joint_state_verified_continuation_pending' if passed else 'failed_audit_preserved'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
