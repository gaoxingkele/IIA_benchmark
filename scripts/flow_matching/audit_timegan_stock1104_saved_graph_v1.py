"""One complete saved-meta-graph observation for the actual Stock1104 mismatch.

TimeGAN, Yoon et al., NeurIPS2019, jsyoon0823/TimeGAN. Frozen TF1.15/Python3.7.
This discriminates graph reconstruction from saved-graph execution; it does not
retry the original failed full45 job or certify all20 evaluator replays.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching import diagnose_timegan_generation_bitwise_v1 as diagnostic

BASE = diagnostic.BASE
ROOT, read, write, sha = BASE.ROOT, BASE.read, BASE.write, BASE.sha


def ancestry(graph_def, root_names):
    nodes = {n.name: n for n in graph_def.node}; pending = list(root_names); seen = {}
    while pending:
        name = pending.pop().split(':')[0].lstrip('^')
        if name in seen:
            continue
        node = nodes[name]; seen[name] = node
        pending.extend(node.input)
    return seen


def node_signature(node):
    # Shape metadata is retained separately; it can differ when importing an
    # original saved graph, and must never silently disappear from evidence.
    return dict(op=node.op, inputs=list(node.input), device=node.device,
        attrs={k: hashlib.sha256(v.SerializeToString(deterministic=True)).hexdigest()
               for k, v in sorted(node.attr.items()) if k != '_output_shapes'})


def graph_difference(source, saved, tensor_name):
    left, right = ancestry(source, [tensor_name]), ancestry(saved, [tensor_name])
    common = sorted(set(left) & set(right))
    changed = [dict(name=n, source=node_signature(left[n]), saved=node_signature(right[n]))
               for n in common if node_signature(left[n]) != node_signature(right[n])]
    shape_changes = [n for n in common if ('_output_shapes' in left[n].attr) != ('_output_shapes' in right[n].attr)
        or ('_output_shapes' in left[n].attr and left[n].attr['_output_shapes'].SerializeToString(deterministic=True)
            != right[n].attr['_output_shapes'].SerializeToString(deterministic=True))]
    return dict(source_forward_nodes=len(left), saved_forward_nodes=len(right),
        source_only_nodes=sorted(set(left)-set(right)), saved_only_nodes=sorted(set(right)-set(left)),
        changed_forward_definitions=changed, shape_metadata_changes=shape_changes,
        forward_definitions_equal_excluding_output_shape_metadata=not changed and set(left)==set(right),
        source_function_library_sha256=hashlib.sha256(source.library.SerializeToString(deterministic=True)).hexdigest(),
        saved_function_library_sha256=hashlib.sha256(saved.library.SerializeToString(deterministic=True)).hexdigest(),
        function_library_bitwise_equal=source.library.SerializeToString(deterministic=True)==saved.library.SerializeToString(deterministic=True))


def verify(config):
    for pin in config['pins']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen saved-graph diagnostic binding changed: ' + pin['path'])
    previous = read(ROOT / config['previous_generation_diagnostic'])
    original, route = diagnostic.verify(previous)
    receipt = read(ROOT / previous['state_root'] / 'receipt.json')
    observed = read(ROOT / previous['output_directory'] / 'generation_difference.json')
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['diagnostic_captured']
            or observed['comparison']['original_bitwise_gate_passed'] is not False
            or observed['comparison']['unequal_coordinates'] != 198
            or config['canonical_job_id'] != route['canonical_job']['id']
            or config['settings'] != previous['settings'] or config['resource_lock'] != previous['resource_lock']
            or config['python'] != previous['python'] or config['cpu_threads'] != 2
            or config['maximum_saved_graph_inference_calls'] != 1
            or config['original_scope'] != previous['original_scope']):
        raise ValueError('Actual failed full generation and exactly one new graph hypothesis required')
    queue, _ = diagnostic.replay.verify(original)
    return queue, route, previous


def worker(config):
    queue, route, previous = verify(config)
    # Retain every full-budget recovered result/artifact/evaluator binding.
    result = diagnostic.replay.completed_native_result(route, queue)
    environment = read(ROOT / queue['environment_lock'])
    for name, key in [('python.exe', 'python_executable_sha256'), ('python37.dll', 'python37_dll_sha256'),
                      ('python37._pth', 'python_pth_sha256')]:
        if sha(Path(environment['runtime_root']) / name) != environment[key]:
            raise ValueError('Frozen numerical interpreter changed')
    import numpy as np
    import tensorflow as tf
    import pkg_resources
    package_root = Path(environment['runtime_root']) / 'Lib/site-packages'
    packages = {p.key: p.version for p in pkg_resources.working_set
                if Path(p.location).resolve() == package_root.resolve()}
    if (Path(sys.executable).resolve() != Path(queue['python']).resolve() or sys.version != environment['python']
            or packages != environment['packages'] or tf.__version__ != '1.15.0' or np.__version__ != '1.18.5'):
        raise ValueError('Exact original numerical environment required')
    from scripts.flow_matching.timegan_native_common_v1 import verify_source, constructor_prefix
    from scripts.flow_matching.timegan_stage_resume_v2 import stochastic_ancestors
    source, manifest = verify_source(read(ROOT / queue['model_config']))
    os.chdir(str(source)); sys.path.insert(0, str(source))
    import timegan as author
    import data_loading as loader
    real, data_audit = BASE.independent_full_loader(route['canonical_job'], read(ROOT / queue['model_config']), np, loader)
    reset = tf.reset_default_graph
    def seeded_reset():
        reset(); tf.set_random_seed(route['canonical_job']['seed'])
    tf.reset_default_graph = seeded_reset
    try:
        nodes = constructor_prefix(author, 'timegan')(real,
            dict(module='gru', hidden_dim=24, num_layer=3, iterations=50000, batch_size=128))
    finally:
        tf.reset_default_graph = reset
    names = {k: nodes[k].name for k in ('X_hat', 'Z', 'X', 'T')}
    source_definition = tf.get_default_graph().as_graph_def(add_shapes=True)
    directory = BASE.checkpoint_directory(result['checkpoints'][2])
    meta = tf.MetaGraphDef(); meta.ParseFromString((directory / 'model.meta').read_bytes())
    comparison = graph_difference(source_definition, meta.graph_def, names['X_hat'])
    imported = tf.Graph()
    with imported.as_default():
        saver = tf.train.import_meta_graph(str(directory / 'model.meta'), clear_devices=False)
        variables = tf.global_variables()
        reader = tf.train.NewCheckpointReader(str(directory / 'model'))
        if {v.op.name: v.shape.as_list() for v in variables} != reader.get_variable_to_shape_map() or len(variables) != 284:
            raise ValueError('All284 saved model and optimizer variables must match')
        tensors = {k: imported.get_tensor_by_name(v) for k, v in names.items()}
        if stochastic_ancestors([tensors['X_hat'].op]):
            raise ValueError('Untracked stochastic inference state forbids this saved graph audit')
        with tf.Session(graph=imported, config=tf.ConfigProto(device_count={'GPU': 0},
                intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)) as session:
            saver.restore(session, str(directory / 'model'))
            before = session.run(variables)
            for v, value in zip(variables, before):
                BASE.bitwise_array(value, reader.get_tensor(v.op.name), np, 'all original saved graph variables')
            native_output = ROOT / route['execution_job']['output_directory']
            with np.load(str(native_output / 'full_native_generation_feed.npz'), allow_pickle=False) as feed:
                for k, field in [('ori_data', 'normalized_real'), ('min_val', 'min_val'), ('max_val', 'max_val')]:
                    BASE.bitwise_array(nodes[k], feed[field], np, 'unchanged full normalization: ' + field)
                BASE.bitwise_array(np.asarray(nodes['ori_time']), feed['times'], np, 'all original full window lengths')
                if list(feed['noise'].shape) != [3661, 24, 6] or not np.isfinite(feed['noise']).all():
                    raise ValueError('Original full3661x24x6 inference feed required')
                actual = session.run(tensors['X_hat'], feed_dict={tensors['Z']: feed['noise'],
                    tensors['X']: feed['normalized_real'], tensors['T']: feed['times']})
                expected = feed['normalized_generated'].copy()
                minimum, scale = feed['min_val'].copy(), feed['max_val'].copy()
            after = session.run(variables)
            for a, b in zip(before, after):
                BASE.bitwise_array(a, b, np, 'no model or optimizer change during one saved graph inference')
    evidence = diagnostic.describe_arrays(actual, expected, np)
    try:
        BASE.bitwise_array(actual, expected, np, diagnostic.DESCRIPTION)
    except ValueError as error:
        evidence.update(original_bitwise_gate_passed=False, original_gate_error=str(error))
    else:
        evidence['original_bitwise_gate_passed'] = True
    with np.load(str(native_output / 'full_native_real_and_generated.npz'), allow_pickle=False) as data:
        renormalized = diagnostic.describe_arrays(actual * scale + minimum, data['generated'], np)
    output = ROOT / config['output_directory']; output.mkdir(parents=True, exist_ok=False)
    arrays = output / 'full_saved_graph_actual_and_expected.npz'
    np.savez_compressed(str(arrays), actual=actual, expected=expected)
    write(output / 'saved_graph_generation_observation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        canonical_job=route['canonical_job'], full_original_data_shape=[3661,24,6], full_input_data_audit=data_audit,
        graph_tensor_names=names, forward_graph_comparison=comparison, generation_comparison=evidence,
        renormalized_comparison=renormalized, full_saved_variables_bitwise_verified=284,
        model_optimizer_state_unchanged_by_inference=True, saved_graph_inference_calls=1,
        full_arrays_sha256=sha(arrays), source_manifest_unchanged=verify_source(read(ROOT / queue['model_config']))[1]==manifest,
        numerical_optimizer_updates_executed=0, additional_training_seed_slots=0, diagnostic_only=True,
        independent_full_checkpoint_replay_completed=False, full20_evaluator_replays_completed=False,
        benchmark_performance=False, all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config); queue, route, previous = verify(config)
    if args.probe_only:
        print(dict(registration_verified=True, full_original_jobs=45, full_input_shape=[3661,24,6],
            maximum_saved_graph_inference_calls=1, tensorflow_imported='tensorflow' in sys.modules,
            numerical_training_executed=False)); return
    if args.worker:
        worker(config); return
    import psutil
    from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None or (base / 'receipt.json').exists():
        raise RuntimeError('Preserve existing saved-graph diagnostic owner or terminal attempt')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=sha(ROOT / args.config), all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            if sha(ROOT / args.config) != state['configuration_sha256']:
                raise ValueError('Live graph hypothesis configuration changed')
            verify(config)
            argv = ['saved_graph_audit', '--config', args.config, '--worker']
            bootstrap = ('import sys,runpy;sys.path.insert(0,' + repr(str(ROOT)) + ');sys.argv=' + repr(argv)
                + ";runpy.run_module('scripts.flow_matching.audit_timegan_stock1104_saved_graph_v1',run_name='__main__')")
            command = [config['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(active_pid=child.pid, active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        target = ROOT / config['output_directory'] / 'saved_graph_generation_observation.json'
        captured = code == 0 and not usage['resource_guard_aborted'] and target.exists()
        write(base / 'receipt.json', dict(exit_code=code, saved_graph_observation_captured=captured,
            independent_full_checkpoint_replay_completed=False, **usage))
        update(dict(active_pid=None, state='saved_graph_observation_captured' if captured else 'failed_audit_preserved'))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
