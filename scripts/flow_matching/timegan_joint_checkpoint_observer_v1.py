"""Save complete native TimeGAN state at source joint-iteration boundaries.

Yoon, Jarrett, van der Schaar, NeurIPS 2019; original jsyoon0823/TimeGAN code.
Only observations and filesystem state are added. Neither optimizer calls nor
random draws nor the full50000 joint loop/all20 evaluators are changed.
"""
import ast
import copy
import inspect
from pathlib import Path
import random

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def equal_numpy_state(left, right, np):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def expected_joint_counts(counts, iteration):
    return (counts['E0'] == 50000 and counts['GS'] == 50000
            and counts['G'] == 2 * iteration and counts['E'] == 2 * iteration
            and 0 <= counts['D'] <= iteration)


def save_joint_boundary(session, itt, nodes, output, config):
    iteration = itt + 1
    interval = config['joint_checkpoint_interval']
    if iteration % interval:
        return
    if interval != 1000 or not 1 <= iteration <= 50000 or session.kind != 'timegan':
        raise ValueError('Only original complete1000-iteration joint boundaries may be saved')
    if not expected_joint_counts(session.counts, iteration):
        raise ValueError('Never save mid-iteration or invented native optimizer counters')
    import numpy as np
    import tensorflow as tf
    from scripts.flow_matching.timegan_stage_resume_v2 import stochastic_ancestors
    roots = [nodes[k + '_solver'] for k in ('E0', 'E', 'D', 'G', 'GS')] + [nodes['X_hat'].op]
    if stochastic_ancestors(roots):
        raise ValueError('Untracked stateful TF training/generation random kernels prohibit exact recovery')
    variables = tf.global_variables()
    if len(variables) != config['checkpoint_full_variable_count']:
        raise ValueError('All native model and optimizer variables must be retained')
    schema = {v.op.name: v.shape.as_list() for v in variables}
    before_np = np.random.get_state()
    before_py = random.getstate()
    before_counts = dict(session.counts)
    old_recording = session.recording
    session.recording = True
    directory = Path(output) / 'recovery_joint_checkpoints' / ('iteration_%06d' % iteration)
    try:
        before_values = session.run(variables)
        if not all(np.isfinite(value).all() for value in before_values):
            raise ValueError('Nonfinite native model/optimizer values')
        directory.mkdir(parents=True, exist_ok=False)
        if session.saver is None:
            session.saver = tf.train.Saver(var_list=variables, max_to_keep=None)
        prefix = directory / 'model'
        session.saver.save(session, str(prefix))
        np.savez_compressed(str(directory / 'numpy_random_state.npz'), algorithm=before_np[0],
            keys=before_np[1], position=before_np[2], has_gauss=before_np[3], cached_gaussian=before_np[4])
        write(directory / 'python_random_state.json', before_py)
        reader = tf.train.NewCheckpointReader(str(prefix))
        if reader.get_variable_to_shape_map() != schema:
            raise ValueError('Complete saved native model/optimizer schema changed')
        if any(not np.array_equal(value, reader.get_tensor(v.op.name)) for v, value in zip(variables, before_values)):
            raise ValueError('Every saved native model/optimizer value must match bitwise')
        after_values = session.run(variables)
        if (schema != {v.op.name: v.shape.as_list() for v in tf.global_variables()}
                or any(not np.array_equal(a, b) for a, b in zip(before_values, after_values))
                or not equal_numpy_state(before_np, np.random.get_state(), np)
                or before_py != random.getstate() or before_counts != session.counts):
            raise ValueError('Observation must preserve all model, optimizer, RNG and update state')
        artifacts = [dict(path=str(p), sha256=sha(p)) for p in sorted(directory.iterdir()) if p.is_file()]
        record = dict(kind='timegan_complete_joint_iteration_checkpoint',
            canonical_job_id=config['canonical_job_id'], completed_joint_iterations=iteration,
            next_original_joint_iteration=iteration, required_joint_iterations=50000,
            optimizer_updates=before_counts, model_parameters=session.parameters,
            native_full_variable_count=len(variables), artifacts=artifacts,
            every_saved_native_model_optimizer_value_bitwise_equal=True,
            model_optimizer_numpy_python_rng_and_counters_unchanged_by_observer=True,
            untracked_training_or_generation_TF_random_operations=[],
            full_source_joint_body_complete=True, additional_optimizer_updates=0,
            additional_training_seed_slots=0, benchmark_performance=False,
            original_native_restore_proof_sha256=config['native_restore_audit_sha256'])
        write(directory / 'receipt.json', record)
        ledger = Path(output) / 'completed_full_joint_checkpoints.json'
        records = read(ledger) if ledger.exists() else []
        if [r['completed_joint_iterations'] for r in records] != list(range(1000, iteration, 1000)):
            raise ValueError('Preserve the full monotonic checkpoint ledger; never replace earlier state')
        records.append(record)
        write(ledger, records)
    finally:
        session.recording = old_recording


def append_joint_observation(tree):
    variant = copy.deepcopy(tree)
    function = variant.body[0]
    loops = [n for n in function.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'itt']
    if len(loops) != 1:
        raise ValueError('Exactly the complete original remaining joint loop is required')
    loops[0].body.append(ast.parse('_save_complete_joint_boundary(sess, itt, locals())').body[0])
    return ast.fix_missing_locations(variant)


def resumed_function_with_checkpoint(module, restore, output, config):
    from scripts.flow_matching.timegan_stage_resume_v2 import resumed_ast
    unit = append_joint_observation(resumed_ast(module))
    def observe(session, itt, nodes):
        save_joint_boundary(session, itt, nodes, output, config)
    namespace = dict(module.__dict__, _verified_restore=restore, _save_complete_joint_boundary=observe)
    exec(compile(unit, module.__file__, 'exec'), namespace)
    return namespace[unit.body[0].name]


def compiler_ast(source):
    tree = ast.parse(source)
    changes = dict(native_function=0, target_observation=0, target_binding=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'resumed_function':
            if ast.dump(node) != ast.dump(ast.parse('resumed_function(module, restore)', mode='eval').body):
                raise ValueError('Only the verified full native resumed function may be observed')
            node.func.id = '_resumed_function_with_checkpoint'
            node.args.extend([ast.Name(id='output', ctx=ast.Load()), ast.Name(id='config', ctx=ast.Load())])
            changes['native_function'] += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'compile':
            if ast.dump(node.args[0]) != ast.dump(ast.parse('ast.fix_missing_locations(tree)', mode='eval').body):
                raise ValueError('The complete legacy observer compiler must be retained')
            node.args[0] = ast.Call(func=ast.Name(id='_corrected_observer_ast', ctx=ast.Load()),
                                   args=[node.args[0]], keywords=[])
            changes['target_observation'] += 1
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'namespace' for t in node.targets):
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != 'dict':
                raise ValueError('The exact observed native namespace must be retained')
            node.value.keywords.append(ast.keyword(arg='_native_predictive_targets',
                value=ast.Name(id='_native_predictive_targets', ctx=ast.Load())))
            changes['target_binding'] += 1
    if any(v != 1 for v in changes.values()):
        raise ValueError('Exactly one joint observer and the existing predictor observation correction are required')
    return ast.fix_missing_locations(tree)


def checkpoint_resume_compiler():
    from scripts.flow_matching import timegan_stage_resume_v2 as original
    from scripts.flow_matching.timegan_predictive_observer_v2 import corrected_observer_ast, native_predictive_targets
    tree = compiler_ast(inspect.getsource(original.compile_observed_resume))
    namespace = dict(original.__dict__, _corrected_observer_ast=corrected_observer_ast,
        _native_predictive_targets=native_predictive_targets,
        _resumed_function_with_checkpoint=resumed_function_with_checkpoint)
    exec(compile(tree, original.__file__, 'exec'), namespace)
    compiled = namespace['compile_observed_resume']
    def compile_bound(config):
        verify_checkpoint_registration(config)
        return compiled(config)
    return compile_bound


def verify_checkpoint_registration(config):
    from scripts.flow_matching.timegan_stage_resume_v2 import binding
    queue, model, job, _ = binding(config)
    proof = read(ROOT / config['native_restore_audit'])
    if (config['joint_checkpoint_interval'] != 1000 or config['joint_iterations'] != 50000
            or config['checkpoint_full_variable_count'] != 284
            or proof['checkpoint_all_variable_restoration']['restored_native_variables'] != 284
            or config['original_training_settings'] != queue['settings']
            or config['full_scope_original_jobs'] != 45
            or config['independent_training_seed_slots_added'] != 0
            or job['id'] != 'timegan_native__gru__sine__seed1103'
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Full source Sine1103 budgets, verified284-variable state and original guards are required')
    return queue, model, job


def verify_checkpointed_recovery_result(config):
    from scripts.flow_matching.timegan_stage_resume_v2 import verify_recovery_result
    result = verify_recovery_result(config)
    records = read(ROOT / config['output_directory'] / 'completed_full_joint_checkpoints.json')
    if [r['completed_joint_iterations'] for r in records] != list(range(1000, 50001, 1000)):
        raise ValueError('All50 complete native checkpoints are required with unchanged full50000 training')
    for record in records:
        if (record['canonical_job_id'] != config['canonical_job_id']
                or not expected_joint_counts(record['optimizer_updates'], record['completed_joint_iterations'])
                or record['native_full_variable_count'] != config['checkpoint_full_variable_count']
                or not record['every_saved_native_model_optimizer_value_bitwise_equal']
                or not record['model_optimizer_numpy_python_rng_and_counters_unchanged_by_observer']
                or record['additional_optimizer_updates'] != 0 or record['additional_training_seed_slots'] != 0):
            raise ValueError('Every full saved checkpoint must preserve source computation and counters')
        for artifact in record['artifacts']:
            if sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Preserve every saved full native state')
    return result
