"""Compare all native restored variables and exact original joint transitions.

Two transition iterations are a recovery verification, not benchmark training.
Both paths use the full original model, full stock data and native optimizer slots.
"""
import argparse
import ast
from datetime import datetime, timezone
import os
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, constructor_prefix
from scripts.flow_matching.timegan_stage_resume_v2 import binding, restore_all, numpy_state


def audit(config):
    queue, model, job, records = binding(config)
    environment = read(ROOT / queue['environment_lock'])
    if sys.version != environment['python']:
        raise ValueError('Frozen original native interpreter required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import tensorflow as tf
    import pkg_resources
    package_root = (ROOT / environment['runtime_root'] / 'Lib/site-packages').resolve()
    actual_packages = {p.key: p.version for p in pkg_resources.working_set if __import__('pathlib').Path(p.location).resolve() == package_root}
    if tf.__version__ != '1.15.0' or actual_packages != environment['packages']:
        raise ValueError('Frozen exact native numerical packages required')
    source = ROOT / model['source_root']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import timegan as author
    from data_loading import real_data_loading
    np.random.seed(job['seed'])
    data = real_data_loading(job['dataset'], model['parameters']['seq_len'])
    shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    if list(np.asarray(data).shape) != shape or not np.isfinite(data).all():
        raise ValueError('Full finite original stock data required')
    native_reset = tf.reset_default_graph

    def reset():
        native_reset()
        tf.set_random_seed(job['seed'])

    tf.reset_default_graph = reset
    parameters = dict(model['parameters'], module=job['module'])
    nodes = constructor_prefix(author, 'timegan')(data, parameters)
    rebuilt_graph = tf.get_default_graph()
    options = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    directory = ROOT / job['output_directory'] / 'checkpoints/supervised_complete'
    checkpoint, rng = directory / 'model', directory / 'numpy_random_state.npz'
    with rebuilt_graph.as_default():
        rebuilt = tf.Session(graph=rebuilt_graph, config=options)
        rebuilt.run(tf.global_variables_initializer())
        restore_proof = restore_all(rebuilt, nodes, checkpoint, rng)
        rebuilt_variables = list(tf.global_variables())
    original_graph = tf.Graph()
    with original_graph.as_default():
        saver = tf.train.import_meta_graph(str(checkpoint) + '.meta', clear_devices=True)
        original = tf.Session(graph=original_graph, config=options)
        saver.restore(original, str(checkpoint))
        original_variables = {v.op.name: v for v in tf.global_variables()}
    if set(original_variables) != {v.op.name for v in rebuilt_variables}:
        raise ValueError('Original saved graph and reconstructed full variable sets differ')
    original_nodes = dict(nodes)
    for key, value in nodes.items():
        if isinstance(value, tf.Tensor):
            original_nodes[key] = original_graph.get_tensor_by_name(value.name)
        elif isinstance(value, tf.Operation):
            original_nodes[key] = original_graph.get_operation_by_name(value.name)
    tree = ast.parse(__import__('pathlib').Path(author.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'timegan')
    joint = [n for n in function.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'itt'][2]
    unit = ast.Module(body=[joint])
    compiled_joint = compile(ast.fix_missing_locations(unit), author.__file__, 'exec')
    final_states = []
    losses = []
    for session, local, variables in [(rebuilt, nodes, rebuilt_variables),
            (original, original_nodes, [original_variables[v.op.name] for v in rebuilt_variables])]:
        np.random.set_state(numpy_state(rng))
        namespace = dict(author.__dict__, **local)
        namespace.update(sess=session, iterations=config['audit_joint_iterations'])
        exec(compiled_joint, namespace)
        final_states.append(session.run(variables))
        losses.append([float(namespace[k]) for k in ['step_e_loss_t0', 'step_g_loss_u', 'step_g_loss_s', 'step_g_loss_v', 'check_d_loss']])
        state = np.random.get_state()
        namespace['_rng_final'] = state
        if session is rebuilt:
            reference_rng = state
        elif not (state[0] == reference_rng[0] and np.array_equal(state[1], reference_rng[1]) and state[2:] == reference_rng[2:]):
            raise ValueError('Original and rebuilt NumPy minibatch/noise trajectories differ')
    equality = all(np.array_equal(a, b) for a, b in zip(*final_states))
    maximum = max(float(np.max(np.abs(a - b))) for a, b in zip(*final_states))
    if not equality or losses[0] != losses[1]:
        raise ValueError('Original saved graph and reconstructed full joint transitions differ')
    variable_records = [dict(name=v.op.name, shape=v.shape.as_list(),
                             final_value_sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest())
                        for v, a in zip(rebuilt_variables, final_states[0])]
    rebuilt.close()
    original.close()
    return dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job_id=job['id'],
        full_original_data_shape=shape, model_parameters=records[-1]['model_parameters'],
        checkpoint_all_variable_restoration=restore_proof, compared_variables=variable_records,
        native_saved_meta_graph_compared=True, full_native_joint_transition_iterations_per_path=config['audit_joint_iterations'],
        all_transition_variable_values_bitwise_equal=equality, maximum_absolute_variable_error=maximum,
        final_losses_original_and_rebuilt=losses, numpy_minibatch_noise_trajectory_identical=True,
        restored_completed_stage_updates=records[-1]['optimizer_updates'],
        verification_only_not_benchmark_training=True, benchmark_performance_rows_added=0,
        full_original_experiments_complete=False, source_sha256=sha(author.__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    # Resolve before the unchanged original data loader's working directory is selected.
    destination = ROOT / config['native_restore_audit']
    if destination.exists():
        raise FileExistsError('Preserve every completed audit attempt')
    write(destination, audit(config))


if __name__ == '__main__':
    main()
