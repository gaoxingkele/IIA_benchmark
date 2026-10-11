"""Full checkpoint/generator/trained evaluator replay for all45 TimeGAN jobs.

Read-only replay restores the full trained critic checkpoints. It executes every
untouched source sampling loop to reconstruct missing test feeds while suppressing
only the unused optimizer/loss fetch. Suppressed calls are protocol iterations,
never training updates. Every full prediction, target, score and post-loop NumPy
state must agree bitwise. No shorter loop, new training or relaxed gate is used.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, verify_counts, verify_queue


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def protocol_contract(kind, source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kind + '_score_metrics']
    if len(functions) != 1:
        raise ValueError('One full untouched original native evaluator function required')
    function = functions[0]
    solver, loss, display, iterations = (('d_solver', 'd_loss', 'step_d_loss', 2000)
        if kind == 'discriminative' else ('p_solver', 'p_loss', 'step_p_loss', 5000))
    # The replacement loss is safe exclusively because it is never consumed.
    uses = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == display]
    if len(uses) != 1 or not isinstance(uses[0].ctx, ast.Store):
        raise ValueError('Suppressed optimizer display loss must be completely unused')
    expected = ast.dump(ast.parse('[%s, %s]' % (solver, loss), mode='eval').body)
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'sess' and n.func.attr == 'run']
    if len(calls) != 3 or sum(bool(c.args) and ast.dump(c.args[0]) == expected for c in calls) != 1:
        raise ValueError('Only original initialization, full solver loop and complete prediction calls permitted')
    assignments = [n for n in function.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'iterations' for t in n.targets)]
    if len(assignments) != 1 or ast.dump(assignments[0].value) != ast.dump(ast.parse(str(iterations), mode='eval').body):
        raise ValueError('All original native evaluator protocol iterations required')
    return dict(kind=kind, iterations=iterations, batch_size=128, solver=solver, loss=loss,
        untouched_function_ast_sha256=hashlib.sha256(ast.dump(function).encode('utf-8')).hexdigest(),
        unused_display_loss_proven=True)


def check_routes(queue, config):
    jobs = {j['id']: j for j in queue['jobs']}
    model = read(ROOT / queue['model_config'])
    datasets = read(ROOT / model['dataset_config'])['datasets']
    routes = config['routes']
    if (len(jobs) != 45 or len(routes) != 45 or len({r['canonical_job']['id'] for r in routes}) != 45
            or {r['canonical_job']['id'] for r in routes} != set(jobs)
            or config['settings'] != queue['settings'] or config['resource_lock'] != queue['resource_lock']
            or config['python'] != queue['python'] or config['cpu_threads'] != 2
            or config['required_checkpoints_each'] != 23 or config['required_evaluators_each'] != 20
            or config['equality_gate'] != 'bitwise' or config['protocol_iterations_each'] != 70000):
        raise ValueError('All45 full original native models/checkpoints/evaluators and unchanged guards required')
    for route in routes:
        canonical = route['canonical_job']
        if canonical != jobs[canonical['id']]:
            raise ValueError('Exact original canonical seed/data/module identity required')
        if route['full_original_data_shape'] != datasets[canonical['dataset']]['full_shape']:
            raise ValueError('Every full original data/window/feature dimension required')
        if route['kind'] == 'original':
            if route['execution_job'] != canonical or route['recovery_configuration'] is not None:
                raise ValueError('Original native job must occupy its exact immutable canonical slot')
        elif route['kind'] == 'stage_resume':
            recovery = read(ROOT / route['recovery_configuration'])
            if recovery['canonical_job'] != canonical or route['execution_job'] != dict(canonical, output_directory=recovery['output_directory']):
                raise ValueError('Only the exact registered same-seed full-stage recovery is permitted')
        else:
            raise ValueError('Unknown native result route')
        if route['result_path'] != (Path(route['execution_job']['output_directory']) / 'result.json').as_posix():
            raise ValueError('Complete actual registered native result path required')
        if not route['native_completion_receipts']:
            raise ValueError('Actual successful guarded native completion receipt required')
    if (config['required_full_generated_windows'] != sum(r['full_original_data_shape'][0] for r in routes)
            or config['required_full_predictive_test_windows'] != 10 * config['required_full_generated_windows']
            or config['required_full_discriminative_test_windows'] != sum(
                20 * (r['full_original_data_shape'][0] - int(r['full_original_data_shape'][0] * .8)) for r in routes)):
        raise ValueError('Full counts across all45 original data tasks required')
    return routes


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete native replay source changed: ' + pin['path'])
    queue = read(ROOT / config['original_queue'])
    verify_queue(queue)
    if config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273):
        raise ValueError('Full original research scope remains required')
    model = read(ROOT / queue['model_config'])
    source = ROOT / model['source_root']
    for kind in ('discriminative', 'predictive'):
        protocol_contract(kind, (source / 'metrics' / (kind + '_metrics.py')).read_text(encoding='utf-8'))
    return queue, check_routes(queue, config)


def checkpoint_directory(checkpoint):
    artifacts = checkpoint['artifacts']
    metas = [Path(a['path']) for a in artifacts if Path(a['path']).name == 'model.meta']
    if len(metas) != 1 or any(Path(a['path']).parent != metas[0].parent for a in artifacts):
        raise ValueError('One exact full native checkpoint directory required')
    names = {Path(a['path']).name for a in artifacts}
    if (not {'model.meta', 'model.index', 'numpy_random_state.npz'}.issubset(names)
            or not any(n.startswith('model.data-') for n in names)):
        raise ValueError('Full native model/meta/optimizer/NumPy state required')
    return metas[0].parent


def completed_native_result(route, queue):
    if route['kind'] == 'original':
        from scripts.flow_matching.run_timegan_native_queue_v1 import verify_result
        return verify_result(queue, route['execution_job'])
    from scripts.flow_matching.timegan_stage_resume_v2 import verify_recovery_result
    return verify_recovery_result(read(ROOT / route['recovery_configuration']))


def successful_native_receipt(route):
    for entry in route['native_completion_receipts']:
        path = ROOT / entry['path']
        if path.exists():
            receipt = read(path)
            if (receipt.get('exit_code') == 0 and receipt.get('resource_guard_aborted') is False
                    and receipt.get(entry['successful_field']) is True):
                return entry, receipt
    return None, None


def complete_metadata(route, queue, result, receipt, receipt_field):
    from scripts.flow_matching.run_timegan_native_job_v1 import validate_evaluators
    model = read(ROOT / queue['model_config'])
    shape = read(ROOT / model['dataset_config'])['datasets'][route['canonical_job']['dataset']]['full_shape']
    verify_counts(result['optimizer_updates'], model)
    validate_evaluators(result['evaluators'])
    tags = ['embedding_complete', 'supervised_complete', 'timegan_complete'] + [
        '%s_%02d_complete' % (kind, repeat) for kind in ('discriminative', 'predictive') for repeat in range(10)]
    if (receipt.get('exit_code') != 0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get(receipt_field) is not True or result.get('job') != route['execution_job']
            or result.get('full_native_three_training_stages_completed') is not True
            or result.get('full_native_evaluator_trainings_completed') is not True
            or result.get('full_original_data_shape') != shape or result.get('full_generated_shape') != shape
            or result.get('independent_model_training_seed_slots') != 1
            or result.get('author_equivalence_certified') is not False or result.get('strict_TAB_result') is not False
            or len(result.get('artifacts', [])) != 22 or len(result.get('visualizations', [])) != 2
            or [c['tag'] for c in result.get('checkpoints', [])] != tags or len(result.get('graph_seeds', [])) != 21):
        raise ValueError('Actual successful full native model/all20 trained evaluators/all23 checkpoints required')
    if route['kind'] == 'original' and result.get('full_original_source_main_completed') is not True:
        raise ValueError('Original complete author main must have executed')
    if route['kind'] == 'stage_resume' and (result.get('source_main_with_verified_stage_resume_completed') is not True
            or result.get('full_original_source_main_completed') is not False):
        raise ValueError('Exact proven prior stage plus full native joint/evaluator completion required')
    proof = read(ROOT / queue['construction_proof'])
    data = route['canonical_job']['dataset']
    model_parameters = next(r['model_parameters'] for r in proof['records']
        if r['dataset'] == data and r['module'] == route['canonical_job']['module'])
    if result['model_parameters'] != model_parameters:
        raise ValueError('Complete original model parameter identity required')
    for index, checkpoint in enumerate(result['checkpoints']):
        checkpoint_directory(checkpoint)
        if index < 3:
            expected = (dict(E0=50000, GS=0, G=0, E=0, D=0) if index == 0 else
                dict(E0=50000, GS=50000, G=0, E=0, D=0) if index == 1 else result['optimizer_updates'])
            if checkpoint['kind'] != 'timegan' or checkpoint['optimizer_updates'] != expected or checkpoint['model_parameters'] != model_parameters:
                raise ValueError('All original complete stage variable/optimizer ledgers required')
        else:
            record = result['evaluators'][index - 3]
            parameters = next(r['parameters'] for r in proof['metric_graphs'] if r['dataset'] == data and r['metric'] == record['kind'])
            if (checkpoint['kind'] != record['kind'] or checkpoint['optimizer_updates'] != {'evaluator': record['optimizer_updates']}
                    or checkpoint['model_parameters'] != parameters):
                raise ValueError('Every full trained evaluator checkpoint must match its original repeat')
    for index, graph in enumerate(result['graph_seeds']):
        kind = 'timegan' if index == 0 else 'discriminative' if index <= 10 else 'predictive'
        repeat = None if index == 0 else (index - 1) % 10
        if graph != dict(index=index, kind=kind, repeat=repeat, seed=route['canonical_job']['seed'] + 1009 * index):
            raise ValueError('Every original graph seed/kind/repeat must match')
    if any(not math.isfinite(result['scores'][kind]) for kind in ('discriminative', 'predictive')):
        raise ValueError('Complete native finite score means required')


def bitwise_array(actual, expected, np, description):
    left, right = np.asarray(actual), np.asarray(expected)
    if (left.shape != right.shape or left.dtype != right.dtype or not np.isfinite(left).all()
            or not np.isfinite(right).all() or left.tobytes() != right.tobytes()):
        raise ValueError('Full finite bitwise equality required: ' + description)
    return hashlib.sha256(left.tobytes()).hexdigest()


def bitwise_score(actual, expected):
    import struct
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)) or struct.pack('!d', float(actual)) != struct.pack('!d', float(expected)):
        raise ValueError('Complete native score must agree bitwise; no relaxed tolerance')


def numpy_state_matches(actual, expected, np):
    import struct
    if (actual[0] != expected[0] or actual[2:4] != expected[2:4]
            or struct.pack('!d', float(actual[4])) != struct.pack('!d', float(expected[4]))
            or not np.array_equal(actual[1], expected[1])):
        raise ValueError('Every complete original post-evaluator NumPy state must match exactly')


def restore_full_checkpoint(session, checkpoint, tf, np):
    variables = tf.global_variables()
    reader = tf.train.NewCheckpointReader(str(checkpoint))
    schema = {v.op.name: v.shape.as_list() for v in variables}
    if schema != reader.get_variable_to_shape_map():
        raise ValueError('Every original native graph variable and optimizer slot required')
    if not any('/Adam' in name for name in schema):
        raise ValueError('Full original trained optimizer state required')
    tf.train.Saver(var_list=variables).restore(session, str(checkpoint))
    values = session.run(variables)
    for variable, value in zip(variables, values):
        bitwise_array(value, reader.get_tensor(variable.op.name), np, 'all model and optimizer variables')
    return dict(full_native_variable_count=len(variables), all_native_variables_restored_bitwise=True,
        schema_sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode('utf-8')).hexdigest())


def independent_full_loader(job, model, np, loader):
    item = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]
    np.random.seed(job['seed'])
    if job['dataset'] == 'sine':
        loaded = loader.sine_data_generation(item['samples'], 24, item['features'])
        independent = []
        rng = np.random.RandomState(job['seed'])
        for _ in range(item['samples']):
            columns = []
            for _ in range(item['features']):
                frequency, phase = rng.uniform(0, .1), rng.uniform(0, .1)
                columns.append([np.sin(frequency * j + phase) for j in range(24)])
            independent.append((np.asarray(columns).T + 1) * .5)
        raw_hash = None
    else:
        raw_path = ROOT / item['raw_file']
        if sha(raw_path) != item['raw_sha256']:
            raise ValueError('Full original preserved raw data changed')
        raw = np.loadtxt(str(raw_path), delimiter=',', skiprows=1)
        if list(raw.shape) != item['raw_shape']:
            raise ValueError('Full original raw shape required')
        loaded = loader.real_data_loading(job['dataset'], 24)
        reverse = raw[::-1]
        normalized = (reverse - reverse.min(0)) / (reverse.max(0) - reverse.min(0) + 1e-7)
        windows = np.asarray([normalized[i:i+24] for i in range(len(raw)-24)])
        independent = windows[np.random.RandomState(job['seed']).permutation(len(windows))]
        raw_hash = sha(raw_path)
    loaded = np.asarray(loaded)
    if list(loaded.shape) != item['full_shape']:
        raise ValueError('Every full original native loader window required')
    digest = bitwise_array(loaded, independent, np, 'independent full native raw/sine loader')
    return loaded, dict(full_shape=list(loaded.shape), raw_sha256=raw_hash,
        independent_full_loader_bitwise_verified=True, native_data_sha256=digest,
        grouped_TAB_splits=False, native_overlapping_random_window_protocol_retained=True)


def check_report(route, binding, report, receipt):
    if (receipt.get('exit_code') != 0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('complete_full_native_replay_passed') is not True
            or report.get('passed') is not True or report.get('canonical_job') != route['canonical_job']
            or report.get('execution_job') != route['execution_job']
            or report.get('result_sha256') != binding['result_sha256']
            or report.get('binding_sha256') != binding['binding_sha256']
            or report.get('complete_checkpoint_restores') != 23 or report.get('complete_trained_evaluator_replays') != 20
            or report.get('original_source_protocol_iterations_reconstructed') != 70000
            or report.get('numerical_optimizer_updates_executed') != 0
            or report.get('training_executed') is not False or report.get('additional_training_seed_slots') != 0
            or report.get('full_original_data_shape') != route['full_original_data_shape']
            or report.get('full_generated_windows_replayed') != route['full_original_data_shape'][0]
            or report.get('complete_normalized_and_renormalized_generation_bitwise_verified') is not True
            or report.get('complete_source_mean_scores_bitwise_verified') is not True
            or report.get('independent_full_loader_bitwise_verified') is not True
            or len(report.get('stage_checkpoint_restores', [])) != 3 or len(report.get('evaluators', [])) != 20):
        raise ValueError('Complete actual full model and all20 trained evaluator replays required')
    if ([s.get('tag') for s in report['stage_checkpoint_restores']] != ['embedding_complete', 'supervised_complete', 'timegan_complete']
            or any(s.get('all_native_variables_restored_bitwise') is not True for s in report['stage_checkpoint_restores'])):
        raise ValueError('All three complete original model/optimizer stage checkpoint restores required')
    for index, row in enumerate(report['evaluators']):
        kind, repeat = ('discriminative' if index < 10 else 'predictive'), index % 10
        if (row.get('kind') != kind or row.get('repeat') != repeat
                or row.get('complete_test_windows') != (route['full_original_data_shape'][0] if kind == 'predictive'
                    else 2 * (route['full_original_data_shape'][0] - int(route['full_original_data_shape'][0] * .8)))
                or row.get('source_protocol_iterations_reconstructed') != (2000 if kind == 'discriminative' else 5000)
                or any(row.get(key) is not True for key in ('all_native_variables_restored_bitwise',
                    'full_native_predictions_bitwise_verified', 'full_native_score_bitwise_verified',
                    'full_post_evaluator_NumPy_state_bitwise_verified', 'no_model_or_optimizer_variables_changed'))):
            raise ValueError('Every complete native checkpoint/feed/protocol/score/state must verify')


def worker(config, route, binding_path):
    queue, _ = verify(config)
    binding = read(ROOT / binding_path)
    if (binding['runtime_configuration_sha256'] != sha(ROOT / config['runtime_configuration'])
            or binding['route'] != route or sha(ROOT / route['result_path']) != binding['result_sha256']
            or sha(ROOT / binding['native_completion_receipt']['path']) != binding['native_receipt_sha256']):
        raise ValueError('Exact frozen actual native completion binding required')
    result = completed_native_result(route, queue)
    native_receipt = read(ROOT / binding['native_completion_receipt']['path'])
    complete_metadata(route, queue, result, native_receipt, binding['native_completion_receipt']['successful_field'])
    environment = read(ROOT / queue['environment_lock'])
    for name, key in [('python.exe', 'python_executable_sha256'), ('python37.dll', 'python37_dll_sha256'), ('python37._pth', 'python_pth_sha256')]:
        if sha(Path(environment['runtime_root']) / name) != environment[key]:
            raise ValueError('Exact original frozen native interpreter required')
    import numpy as np
    import tensorflow as tf
    import pkg_resources
    package_root = Path(environment['runtime_root']) / 'Lib/site-packages'
    packages = {p.key: p.version for p in pkg_resources.working_set if Path(p.location).resolve() == package_root.resolve()}
    if (Path(sys.executable).resolve() != Path(queue['python']).resolve() or sys.version != environment['python']
            or packages != environment['packages'] or tf.__version__ != '1.15.0' or np.__version__ != '1.18.5'):
        raise ValueError('Exact complete author native numerical environment required')
    model = read(ROOT / queue['model_config'])
    from scripts.flow_matching.timegan_native_common_v1 import verify_source, constructor_prefix
    from scripts.flow_matching.timegan_stage_resume_v2 import numpy_state, stochastic_ancestors
    source, before = verify_source(model)
    os.chdir(str(source)); sys.path.insert(0, str(source))
    import timegan as author_model
    import data_loading as loader
    from metrics import discriminative_metrics as disc, predictive_metrics as pred
    if any(Path(module.__file__).resolve().parent not in (source.resolve(), (source / 'metrics').resolve())
            for module in (author_model, loader, disc, pred)):
        raise ValueError('Only the exact untouched original author modules required')
    output = ROOT / config['output_root'] / route['canonical_job']['id']
    if output.exists():
        raise FileExistsError('Preserve every original complete or failed native replay')
    output.mkdir(parents=True)
    real, data_audit = independent_full_loader(route['canonical_job'], model, np, loader)
    native_output = ROOT / route['execution_job']['output_directory']
    with np.load(str(native_output / 'full_native_real_and_generated.npz'), allow_pickle=False) as archive:
        bitwise_array(real, archive['real'], np, 'all original real windows')
        generated = archive['generated'].copy()
    if generated.shape != real.shape or not np.isfinite(generated).all():
        raise ValueError('Complete original finite generated windows required')
    original_reset, original_session = tf.reset_default_graph, tf.Session
    graph_seeds = []
    context = dict(kind='timegan', repeat=None, session=None, checkpoint=None)

    def reset():
        original_reset()
        index = len(graph_seeds)
        seed = route['canonical_job']['seed'] + 1009 * index
        tf.set_random_seed(seed)
        graph_seeds.append(dict(index=index, kind=context['kind'], repeat=context['repeat'], seed=seed))

    tf.reset_default_graph = reset
    nodes = constructor_prefix(author_model, 'timegan')(real, dict(module=route['canonical_job']['module'],
        hidden_dim=24, num_layer=3, iterations=50000, batch_size=128))
    session_config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
    stage_rows = []
    with original_session(config=session_config) as session:
        for checkpoint in result['checkpoints'][:3]:
            directory = checkpoint_directory(checkpoint)
            evidence = restore_full_checkpoint(session, directory / 'model', tf, np)
            stage_rows.append(dict(tag=checkpoint['tag'], **evidence))
        if stochastic_ancestors([nodes['X_hat'].op]):
            raise ValueError('Generator checkpoint lacks state for stochastic inference ancestors')
        with np.load(str(native_output / 'full_native_generation_feed.npz'), allow_pickle=False) as feed:
            bitwise_array(nodes['ori_data'], feed['normalized_real'], np, 'full author model normalization')
            bitwise_array(nodes['min_val'], feed['min_val'], np, 'original full min values')
            bitwise_array(nodes['max_val'], feed['max_val'], np, 'original full max values')
            times = np.asarray(nodes['ori_time'])
            bitwise_array(times, feed['times'], np, 'every original window length')
            if feed['noise'].shape != real.shape or not np.isfinite(feed['noise']).all():
                raise ValueError('Every original complete generation noise sample required')
            calculated = session.run(nodes['X_hat'], feed_dict={nodes['Z']: feed['noise'], nodes['X']: nodes['ori_data'], nodes['T']: times})
            generation_digest = bitwise_array(calculated, feed['normalized_generated'], np, 'complete trained normalized generation')
            bitwise_array(calculated * nodes['max_val'] + nodes['min_val'], generated, np, 'complete author generated renormalization')
        # Inference must preserve every trained model/optimizer value.
        reader = tf.train.NewCheckpointReader(str(checkpoint_directory(result['checkpoints'][2]) / 'model'))
        for variable, value in zip(tf.global_variables(), session.run(tf.global_variables())):
            bitwise_array(value, reader.get_tensor(variable.op.name), np, 'unchanged full model and optimizer after generation')
    np.random.set_state(numpy_state(checkpoint_directory(result['checkpoints'][2]) / 'numpy_random_state.npz'))
    del nodes, calculated, reader
    gc.collect()

    class ReplayedSession(original_session):
        def __init__(self, *args, **kwargs):
            self.nodes = dict(inspect.currentframe().f_back.f_locals)
            self.kind, self.repeat = context['kind'], context['repeat']
            self.contract = protocol_contract(self.kind, inspect.getsource(disc if self.kind == 'discriminative' else pred))
            self.iterations, self.predictions = 0, 0
            self.internal, self.restored = False, False
            self.evidence = None
            inference_ops = ([self.nodes['y_pred_real'].op, self.nodes['y_pred_fake'].op]
                if self.kind == 'discriminative' else [self.nodes['y_pred'].op])
            if stochastic_ancestors(inference_ops):
                raise ValueError('Trained evaluator checkpoint lacks stochastic inference state')
            kwargs['config'] = session_config
            super(ReplayedSession, self).__init__(*args, **kwargs)
            context['session'] = self

        def run(self, fetches, *args, **kwargs):
            if self.internal:
                return super(ReplayedSession, self).run(fetches, *args, **kwargs)
            if not self.restored:
                if not isinstance(fetches, tf.Operation) or fetches.type != 'NoOp' or fetches.name != 'init':
                    raise ValueError('Only original global initialization may trigger complete checkpoint restoration')
                self.internal = True
                try:
                    self.evidence = restore_full_checkpoint(self, checkpoint_directory(context['checkpoint']) / 'model', tf, np)
                finally:
                    self.internal = False
                self.restored = True
                return None
            solver, loss = self.nodes[self.contract['solver']], self.nodes[self.contract['loss']]
            if isinstance(fetches, list) and len(fetches) == 2 and fetches[0] is solver and fetches[1] is loss:
                self.iterations += 1
                if self.iterations > self.contract['iterations']:
                    raise ValueError('Unexpected original source protocol iteration count')
                feed = kwargs['feed_dict']
                if any(np.asarray(value).shape[0] != 128 for value in feed.values()):
                    raise ValueError('Every full native source minibatch draw required')
                if self.iterations % 1000 == 0:
                    write(output / 'progress.json', dict(current_kind=self.kind, current_repeat=self.repeat,
                        source_protocol_iterations_reconstructed=self.iterations, numeric_optimizer_updates=0,
                        required_original_jobs=45, required_evaluator_replays_each=20, all_original_experiments_complete=False))
                # Both original display losses are statically proven unused.
                return [None, 0.0]
            expected = ([self.nodes['y_pred_real'], self.nodes['y_pred_fake']]
                if self.kind == 'discriminative' else self.nodes['y_pred'])
            exact = (isinstance(fetches, list) and len(fetches) == 2
                and all(a is b for a, b in zip(fetches, expected))) if self.kind == 'discriminative' else fetches is expected
            if not exact or self.iterations != self.contract['iterations'] or self.predictions != 0:
                raise ValueError('Exactly one complete prediction after every original protocol iteration required')
            self.predictions += 1
            values = super(ReplayedSession, self).run(fetches, *args, **kwargs)
            archive_path = native_output / ('%s_%02d_full_test.npz' % (self.kind, self.repeat))
            with np.load(str(archive_path), allow_pickle=False) as archive:
                if self.kind == 'discriminative':
                    count = len(real) - int(len(real) * .8)
                    if any(np.asarray(value).shape != (count, 1) for value in values):
                        raise ValueError('Both complete original 20percent discriminator test groups required')
                    digests = [bitwise_array(values[0], archive['real_probabilities'], np, 'complete real discriminator probabilities'),
                        bitwise_array(values[1], archive['synthetic_probabilities'], np, 'complete synthetic discriminator probabilities')]
                    tested_points = 2 * count
                else:
                    from scripts.flow_matching.timegan_predictive_observer_v2 import native_predictive_targets
                    targets = native_predictive_targets(kwargs['feed_dict'], self.nodes['Y'],
                        inspect.currentframe().f_back.f_locals, values, np)
                    if np.asarray(values).shape != (len(real), 23, 1):
                        raise ValueError('Every original predictive window and all23 target steps required')
                    digests = [bitwise_array(values, archive['predictions'], np, 'complete trained predictor output'),
                        bitwise_array(targets, archive['targets'], np, 'every original native prediction target')]
                    tested_points = len(real)
            self.evidence.update(full_native_predictions_bitwise_verified=True, full_prediction_array_sha256=digests,
                complete_test_windows=tested_points, complete_test_feed_sha256=[
                    hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for value in kwargs['feed_dict'].values()])
            return values

        def verify_unchanged(self):
            self.internal = True
            try:
                reader = tf.train.NewCheckpointReader(str(checkpoint_directory(context['checkpoint']) / 'model'))
                for variable, value in zip(tf.global_variables(), self.run(tf.global_variables())):
                    bitwise_array(value, reader.get_tensor(variable.op.name), np, 'trained critic and optimizer never updated')
            finally:
                self.internal = False

    tf.Session = ReplayedSession
    evaluator_rows, scores = [], dict(discriminative=[], predictive=[])
    for index, record in enumerate(result['evaluators']):
        context.update(kind=record['kind'], repeat=record['repeat'], checkpoint=result['checkpoints'][index + 3], session=None)
        function = disc.discriminative_score_metrics if record['kind'] == 'discriminative' else pred.predictive_score_metrics
        score = float(function(real, generated))
        session = context['session']
        if session.iterations != (2000 if record['kind'] == 'discriminative' else 5000) or session.predictions != 1:
            raise ValueError('Every original source draw and exactly one full trained prediction required')
        bitwise_score(score, record['score'])
        numpy_state_matches(np.random.get_state(), numpy_state(checkpoint_directory(context['checkpoint']) / 'numpy_random_state.npz'), np)
        session.verify_unchanged()
        row = dict(kind=record['kind'], repeat=record['repeat'], score=score, **session.evidence)
        row.update(source_protocol_iterations_reconstructed=session.iterations, numerical_optimizer_updates_executed=0,
            full_native_score_bitwise_verified=True, full_post_evaluator_NumPy_state_bitwise_verified=True,
            no_model_or_optimizer_variables_changed=True, original_graph_seed=result['graph_seeds'][index + 1]['seed'])
        evaluator_rows.append(row); scores[record['kind']].append(score)
        session.close(); context['session'] = None
        gc.collect()
    if graph_seeds != result['graph_seeds']:
        raise ValueError('All21 original model/evaluator graph seeds must match')
    for kind, values in scores.items():
        bitwise_score(float(np.mean(values)), result['scores'][kind])
    tf.reset_default_graph, tf.Session = original_reset, original_session
    if verify_source(model)[1] != before or sha(ROOT / route['result_path']) != binding['result_sha256']:
        raise ValueError('Original source or actual completed native result changed during replay')
    if sha(ROOT / binding['native_completion_receipt']['path']) != binding['native_receipt_sha256']:
        raise ValueError('Original actual full completion receipt changed')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        canonical_job=route['canonical_job'], execution_job=route['execution_job'], binding_sha256=sha(ROOT / binding_path),
        result_sha256=binding['result_sha256'], full_original_data_shape=list(real.shape),
        full_generated_windows_replayed=len(real), generation_array_sha256=generation_digest,
        complete_normalized_and_renormalized_generation_bitwise_verified=True,
        complete_checkpoint_restores=23, stage_checkpoint_restores=stage_rows,
        complete_trained_evaluator_replays=20, evaluators=evaluator_rows,
        original_source_protocol_iterations_reconstructed=70000, numerical_optimizer_updates_executed=0,
        independent_full_loader_bitwise_verified=True, data_audit=data_audit,
        complete_source_mean_scores_bitwise_verified=True, training_executed=False,
        additional_training_seed_slots=0, author_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False)
    write(output / 'complete_full_native_replay.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--canonical-id'); parser.add_argument('--binding')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if config['runtime_configuration'] != args.config:
        raise ValueError('Exact immutable full45 replay binding required')
    queue, routes = verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, required_original_jobs=45, required_checkpoints=1035,
            required_trained_evaluator_replays=900, required_full_generated_windows=config['required_full_generated_windows'],
            source_protocol_iterations_each=70000, training_executed=False,
            tensorflow_imported='tensorflow' in sys.modules, numpy_imported='numpy' in sys.modules,
            all_original_experiments_complete=False)))
        return
    if args.worker:
        if not args.canonical_id or not args.binding:
            raise ValueError('Exact complete native case binding required')
        worker(config, next(r for r in routes if r['canonical_job']['id'] == args.canonical_id), args.binding)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Complete full45 native replay controller already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=sha(ROOT / args.config), required_original_models=45,
        verified_original_ids=[], failed={}, pending_original_models=45, all_original_experiments_complete=False)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)

    try:
        while len(state['verified_original_ids']) < 45:
            for route in routes:
                identity = route['canonical_job']['id']
                if identity in state['verified_original_ids'] or identity in state['failed']:
                    continue
                if not (ROOT / route['result_path']).exists():
                    continue
                entry, receipt = successful_native_receipt(route)
                if entry is None:
                    continue
                try:
                    result = completed_native_result(route, queue)
                    complete_metadata(route, queue, result, receipt, entry['successful_field'])
                    binding = dict(route=route, runtime_configuration_sha256=sha(ROOT / args.config),
                        result_sha256=sha(ROOT / route['result_path']), native_completion_receipt=entry,
                        native_receipt_sha256=sha(ROOT / entry['path']))
                    binding_path = base / 'bindings' / (identity + '.json'); exclusive(binding_path, binding)
                    argv = ['native_full45_replay', '--config', args.config, '--worker', '--canonical-id', identity, '--binding', str(binding_path)]
                    bootstrap = ("import sys,runpy;sys.path.insert(0," + repr(str(ROOT)) + ");sys.argv=" + repr(argv) + ";"
                        "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v1',run_name='__main__')")
                    command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
                    with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                        logs = base / 'logs'; logs.mkdir(exist_ok=True)
                        with (logs / (identity + '.stdout.log')).open('xb') as out, (logs / (identity + '.stderr.log')).open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            update(dict(state='running_complete_full45_native_case_replay', active_pid=child.pid,
                                active_create_time=psutil.Process(child.pid).create_time(), active_job=identity, actual_command=command))
                            code, usage = capped_wait(child, config, api, update)
                    report_path = ROOT / config['output_root'] / identity / 'complete_full_native_replay.json'
                    passed = code == 0 and usage['resource_guard_aborted'] is False and report_path.exists()
                    final_receipt = dict(exit_code=code, complete_full_native_replay_passed=passed, **usage)
                    write(base / 'receipts' / (identity + '.json'), final_receipt)
                    if not passed:
                        raise ValueError('Actual complete trained-model/evaluator native replay failed; preserve its full attempt')
                    binding['binding_sha256'] = sha(binding_path)
                    check_report(route, binding, read(report_path), final_receipt)
                    state['verified_original_ids'].append(identity)
                except Exception as error:
                    state['failed'][identity] = str(error)
                    exclusive(base / 'failures' / (identity + '.json'), dict(error=str(error), route=route, all_original_experiments_complete=False))
                update(dict(active_pid=None, active_job=None, pending_original_models=45-len(state['verified_original_ids'])))
            owners = []
            for expected in config['expected_original_native_owners']:
                try:
                    process = psutil.Process(expected['pid'])
                    if process.create_time() == expected['create_time'] and process.cmdline() == expected['command']:
                        owners.append(dict(expected, live_now=True))
                except psutil.NoSuchProcess:
                    pass
            update(dict(state='waiting_actual_complete_original_full45_native_results', actual_verified_live_original_owners=owners))
            if len(state['verified_original_ids']) < 45:
                time.sleep(config['poll_seconds'])
        update(dict(state='all45_complete_native_model_and_evaluator_replays_verified', all45_TimeGAN_native_replays_complete=True))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
