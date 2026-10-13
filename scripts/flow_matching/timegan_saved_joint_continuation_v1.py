"""Continue actual Sine15000 state through all50000 joint iterations/all20 evals.

Yoon, Jarrett, van der Schaar, TimeGAN, NeurIPS2019; jsyoon0823/TimeGAN.
Python3.7 native compatible. Never modify frozen author or live worker files.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching import timegan_stage_resume_v2 as stages
from scripts.flow_matching import timegan_joint_checkpoint_observer_v1 as observer
from scripts.flow_matching.audit_timegan_sine_saved_joint15000_v1 import tuple_state


def registration(config):
    from scripts.flow_matching.audit_timegan_sine_saved_joint15000_v1 import verify
    for pin in config['continuation_source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen continuation source changed: ' + pin['path'])
    audit_config = read(ROOT / config['saved_joint_native_audit_configuration'])
    queue, model, job, saved = verify(audit_config)
    previous = read(ROOT / audit_config['previous_recovery_configuration'])
    if (config['canonical_job'] != job or config['settings'] != audit_config['settings']
            or config['original_scope'] != audit_config['original_scope']
            or config['remaining_joint_iterations'] != 35000
            or config['saved_joint_iterations'] != 15000
            or config['joint_iterations'] != 50000
            or config['joint_checkpoint_interval'] != 1000
            or config['checkpoint_full_variable_count'] != 284
            or config['independent_training_seed_slots_added'] != 0
            or config['original_training_settings'] != queue['settings']
            or config['original_queue_sha256'] != previous['original_queue_sha256']):
        raise ValueError('Original full scope/budgets and stronger admission only required')
    stages.binding(config)
    forbidden = [job['output_directory'], previous['output_directory']]
    if (ROOT / config['output_directory']).resolve() in [(ROOT / p).resolve() for p in forbidden]:
        raise ValueError('Preserve original and interrupted saved-state attempts')
    return queue, model, job, saved, audit_config


def require_native_proof(config, expected_sha=None):
    queue, model, job, saved, audit = registration(config)
    path = ROOT / audit['output_directory'] / 'saved_joint15000_native_restore.json'
    guard = read(ROOT / audit['state_root'] / 'receipt.json')
    proof = read(path)
    digest = sha(path)
    restore = proof['native_restore']
    directory = (ROOT / audit['joint_checkpoint_receipt']).parent
    expected_shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    if (expected_sha is not None and digest != expected_sha
            or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['full_native_joint15000_restore_passed'] is not True
            or proof['passed'] is not True or proof['canonical_job_id'] != job['id']
            or proof['saved_optimizer_updates'] != saved['optimizer_updates']
            or proof['completed_joint_iterations'] != 15000
            or proof['remaining_original_joint_iterations'] != 35000
            or proof['required_original_joint_iterations'] != 50000
            or proof['full_native_parameters'] != saved['model_parameters']
            or proof['full_original_data_shape'] != expected_shape
            or restore['checkpoint_index_sha256'] != sha(directory / 'model.index')
            or restore['rng_sha256'] != sha(directory / 'numpy_random_state.npz')
            or restore['restored_native_variables'] != 284
            or restore['all_variable_values_bitwise_equal'] is not True
            or restore['all_variables_finite'] is not True
            or restore['stochastic_training_or_generation_ancestors']
            or proof['native_NumPy_and_Python_RNG_bitwise_restored'] is not True
            or proof['new_optimizer_updates_executed'] != 0
            or proof['additional_training_seed_slots'] != 0):
        raise ValueError('Actual guarded complete native state restoration required')
    return saved, audit, digest


def joint_ast(module):
    unit = stages.resumed_ast(module)
    loop = next(n for n in unit.body[0].body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'itt')
    expected = ast.parse('range(iterations)', mode='eval').body
    if ast.dump(loop.iter) != ast.dump(expected):
        raise ValueError('Exact author joint iteration range required')
    loop.iter = ast.parse('range(15000, iterations)', mode='eval').body
    # The original joint body is untouched; observation comes after its full body.
    return observer.append_joint_observation(ast.fix_missing_locations(unit))


def bind_joint_function(module, restore, output, config):
    saved, audit, digest = require_native_proof(config, config['_native_proof_sha256'])
    prior_records = read(ROOT / audit['completed_joint_ledger'])
    directory = (ROOT / audit['joint_checkpoint_receipt']).parent

    def restore_saved(session, nodes):
        restore(session, nodes)
        expected_py = tuple_state(read(directory / 'python_random_state.json'))
        random.setstate(expected_py)
        if random.getstate() != expected_py or session.counts != saved['optimizer_updates']:
            raise ValueError('Saved Python RNG and cumulative optimizer counts required')
        evidence = read(output / 'resume_restore_verification.json')
        evidence.update(saved_joint_iterations=15000, remaining_joint_iterations=35000,
            Python_RNG_bitwise_restored=True, saved_joint_native_proof_sha256=digest)
        write(output / 'resume_restore_verification.json', evidence)
        ledger = output / 'completed_full_joint_checkpoints.json'
        if ledger.exists():
            raise FileExistsError('Never replace a continuation checkpoint ledger')
        write(ledger, prior_records)

    def observe(session, itt, nodes):
        observer.save_joint_boundary(session, itt, nodes, output, config)

    namespace = dict(module.__dict__, _verified_restore=restore_saved,
                     _save_complete_joint_boundary=observe)
    unit = joint_ast(module)
    exec(compile(unit, module.__file__, 'exec'), namespace)
    return namespace[unit.body[0].name]


def compiler_tree():
    tree = observer.compiler_ast(inspect.getsource(stages.compile_observed_resume))
    changed = dict(prior=0, directory=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'prior':
                node.value = ast.Name(id='_saved_prior', ctx=ast.Load()); changed['prior'] += 1
            if node.targets[0].id == 'directory':
                node.value = ast.Name(id='_saved_directory', ctx=ast.Load()); changed['directory'] += 1
    if changed != dict(prior=1, directory=1):
        raise ValueError('Exactly one saved-state path and counter binding required')
    return ast.fix_missing_locations(tree)


def compile_continuation(config, proof_sha):
    from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_observer_ast, native_predictive_targets
    saved, audit, digest = require_native_proof(config, proof_sha)
    effective = dict(config, _native_proof_sha256=digest)
    namespace = dict(stages.__dict__, _saved_prior=saved['optimizer_updates'],
        _saved_directory=(ROOT / audit['joint_checkpoint_receipt']).parent,
        _resumed_function_with_checkpoint=bind_joint_function,
        _corrected_observer_ast=corrected_observer_ast,
        _native_predictive_targets=native_predictive_targets)
    exec(compile(compiler_tree(), __file__, 'exec'), namespace)
    function, queue, job = namespace['compile_observed_resume'](effective)

    def run_complete(verified_queue, recovered_job, queue_path):
        function(verified_queue, recovered_job, queue_path)
        path = ROOT / config['output_directory'] / 'result.json'
        result = read(path)
        result.update(saved_complete_joint_iterations=15000, fresh_complete_joint_iterations=35000,
            required_complete_joint_iterations=50000, saved_joint_native_proof_sha256=digest,
            recovery_route='verified_saved_joint_continuation', unsaved_previous_joint_iterations_credited=0)
        write(path, result)

    return run_complete, queue, job


def verify_result(config, proof_sha):
    saved, audit, digest = require_native_proof(config, proof_sha)
    effective = dict(config, _native_proof_sha256=digest)
    # Retain every original full result/evaluator/artifact gate, changing only
    # the explicitly attributed prior optimizer counts from supervised to15000.
    tree = ast.parse(inspect.getsource(stages.verify_recovery_result))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'prior':
            node.value = ast.Name(id='_saved_prior', ctx=ast.Load()); changed += 1
    if changed != 1:
        raise ValueError('All original completion gates must remain')
    namespace = dict(stages.__dict__, _saved_prior=saved['optimizer_updates'])
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    result = namespace['verify_recovery_result'](effective)
    if (result['saved_complete_joint_iterations'] != 15000 or result['fresh_complete_joint_iterations'] != 35000
            or result['required_complete_joint_iterations'] != 50000
            or result['saved_joint_native_proof_sha256'] != digest
            or result['unsaved_previous_joint_iterations_credited'] != 0
            or not 0 <= result['fresh_optimizer_updates']['D'] <= 35000):
        raise ValueError('Exact saved15000 plus fresh35000 attribution required')
    output = ROOT / config['output_directory']
    evidence = read(output / 'resume_restore_verification.json')
    if (evidence['restored_native_variables'] != 284 or not evidence['all_variable_values_bitwise_equal']
            or not evidence['all_variables_finite'] or not evidence['Python_RNG_bitwise_restored']
            or evidence['saved_joint_native_proof_sha256'] != digest
            or evidence['restored_completed_stage_updates'] != saved['optimizer_updates']):
        raise ValueError('Actual continuation restore evidence required')
    records = read(output / 'completed_full_joint_checkpoints.json')
    if ([r['completed_joint_iterations'] for r in records] != list(range(1000, 50001, 1000))
            or records[:15] != read(ROOT / audit['completed_joint_ledger'])):
        raise ValueError('Preserve15 real states and complete35 remaining checkpoints')
    for r in records:
        if (r['canonical_job_id'] != config['canonical_job_id']
                or not observer.expected_joint_counts(r['optimizer_updates'], r['completed_joint_iterations'])
                or r['native_full_variable_count'] != 284
                or r['every_saved_native_model_optimizer_value_bitwise_equal'] is not True
                or r['model_optimizer_numpy_python_rng_and_counters_unchanged_by_observer'] is not True
                or r['untracked_training_or_generation_TF_random_operations']
                or r['full_source_joint_body_complete'] is not True
                or r['additional_optimizer_updates'] != 0 or r['additional_training_seed_slots'] != 0):
            raise ValueError('Complete native boundary fidelity required')
        for artifact in r['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Saved full joint artifact changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--audit-proof-sha256')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config)
    queue, model, job, saved, audit = registration(config)
    if args.probe_only:
        print(dict(original_jobs=len(queue['jobs']), saved_joint_iterations=15000,
            remaining_joint_iterations=35000, all20_evaluators_required=True,
            native_audit_required_before_training=True, tensorflow_imported='tensorflow' in sys.modules))
        return
    if args.worker:
        if not args.audit_proof_sha256:
            raise ValueError('Frozen actual native audit digest required')
        function, verified_queue, recovered_job = compile_continuation(config, args.audit_proof_sha256)
        function(verified_queue, recovered_job, config['original_queue'])
        verify_result(config, args.audit_proof_sha256); return
    import psutil
    from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Continuation already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=sha(ROOT / args.config), all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)

    try:
        if (ROOT / config['output_directory']).exists():
            raise FileExistsError('Preserve prior continuation attempts')
        while not (ROOT / audit['state_root'] / 'receipt.json').exists():
            update(dict(state='waiting_for_actual_native_joint15000_audit', shared_lock_owned=False))
            time.sleep(15)
        saved, audit, digest = require_native_proof(config)
        with api['resource_slot'](ROOT / queue['resource_lock'], config['settings'], update):
            if sha(ROOT / args.config) != state['configuration_sha256']:
                raise ValueError('Live continuation configuration changed')
            require_native_proof(config, digest)
            write(base / 'dispatch_binding.json', dict(configuration_sha256=state['configuration_sha256'],
                actual_native_audit_sha256=digest, saved_joint_iterations=15000,
                remaining_joint_iterations=35000, original_full_joint_iterations=50000))
            argv = ['saved_joint_continuation', '--config', args.config, '--worker', '--audit-proof-sha256', digest]
            bootstrap = ('import sys,runpy;sys.path.insert(0,' + repr(str(ROOT)) + ');sys.argv=' + repr(argv)
                + ";runpy.run_module('scripts.flow_matching.timegan_saved_joint_continuation_v1',run_name='__main__')")
            command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                        OMP_NUM_THREADS=str(queue['cpu_threads']), MKL_NUM_THREADS=str(queue['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_remaining35000_and_all20_evaluators', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, dict(queue, settings=config['settings']), api, update)
        passed = False
        if code == 0 and not usage['resource_guard_aborted']:
            verify_result(config, digest); passed = True
        write(base / 'receipt.json', dict(exit_code=code, full_native_recovery_result_verified=passed,
            independent_full45_replay_completed=False, **usage))
        update(dict(active_pid=None, state='completed_requires_independent_replay' if passed else 'failed_attempt_preserved'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
