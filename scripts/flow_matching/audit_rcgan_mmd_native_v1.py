"""Audit complete original MNIST evaluator and full sigma stopping budget.

Full original model/data/evaluator preparation only: no GAN training or scores
are published as benchmark performance. All author assets remain read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
from scripts.flow_matching.rcgan_mmd_runtime_v1 import cached_kernel, repair_proven_zero_distances


def audit(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full native evaluator registration changed')
    registry = read(ROOT / config['source_registry'])
    asset = next(a for a in registry['assets'] if a['id'] == 'rcgan')
    before = verify_source(asset)
    if Path(sys.executable).resolve() != Path(config['python']).resolve():
        raise ValueError('Registered native interpreter required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import tensorflow as tf
    if tf.__version__ != '1.15.0' or np.__version__ != '1.18.5':
        raise ValueError('Frozen native numerical versions required')
    sys.path.insert(0, asset['original_directory'])
    import model
    import utils
    import data_utils
    import mmd
    np.random.seed(config['seed'])
    tf.set_random_seed(config['seed'])
    settings = vars(utils.rgan_options_parser().parse_args([]))
    settings.update(read(Path(asset['original_directory']) / config['source_settings']))
    if settings['num_epochs'] != 500 or settings['batch_size'] != 28 or settings['D_rounds'] != 1 or settings['G_rounds'] != 1:
        raise ValueError('Full original MNIST500/D1/G1/batch28 settings required')
    dataset = read(ROOT / config['dataset_configuration'])
    if sha(dataset['derived_training_array']) != config['derived_training_sha256']:
        raise ValueError('Original complete60000 image table changed')
    os.chdir(config['working_directory'])
    samples, pdf, labels = data_utils.get_samples_and_labels(settings)
    if {k: list(v.shape) for k, v in samples.items()} != config['full_split_shapes']:
        raise ValueError('Original complete36000/12000/12000 split required')
    with np.load(config['prior_split_indices'], allow_pickle=False) as indices:
        table = np.load(dataset['derived_training_array'], mmap_mode='r', allow_pickle=False)
        for tag in ('train', 'vali', 'test'):
            for offset in range(0, len(samples[tag]), 1000):
                selected = indices[tag][offset:offset + 1000]
                if (not np.array_equal(samples[tag][offset:offset + 1000].reshape(-1, 784), table[selected, 1:])
                        or not np.array_equal(labels[tag][offset:offset + 1000], table[selected, 0])):
                    raise ValueError('Every original split pixel/label must match source image identities')
    print('full_original_60000_image_split_verified', flush=True)
    original_sqrt = np.sqrt
    distance_record = {}

    def observed_sqrt(value, *args, **kwargs):
        distance_record['squared'] = value
        return original_sqrt(value, *args, **kwargs)

    started = time.perf_counter()
    np.sqrt = observed_sqrt
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            original_bandwidth = float(mmd.median_pairwise_distance(samples['vali']))
    finally:
        np.sqrt = original_sqrt
    squared = distance_record.pop('squared')
    if squared.shape != (12000, 12000):
        raise ValueError('All144000000 original validation pair distances required')
    distance_digest = hashlib.sha256()
    for offset in range(0, len(squared), 128):
        distance_digest.update(squared[offset:offset + 128].tobytes())
    original_squared_sha256 = distance_digest.hexdigest()
    repair = repair_proven_zero_distances(squared, samples['vali'], samples['vali'])
    # The original np.median includes diagonal pairs; retain all144M entries.
    np.sqrt(squared, out=squared)
    bandwidth = float(np.median(squared, overwrite_input=True))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Full corrected original bandwidth must be finite positive')
    if np.isfinite(original_bandwidth) and original_bandwidth != bandwidth:
        raise ValueError('Finite original full bandwidth cannot be changed')
    median_evidence = dict(original_bandwidth=original_bandwidth if np.isfinite(original_bandwidth) else None,
        original_bandwidth_finite=bool(np.isfinite(original_bandwidth)), corrected_full_bandwidth=bandwidth,
        full_original_pairs=144000000, original_squared_distances_sha256=original_squared_sha256,
        original_warnings=[str(w.message) for w in caught], elapsed_seconds=time.perf_counter() - started, **repair)
    del squared
    write(ROOT / config['output_directory'] / 'full_pairwise_bandwidth_audit.json', median_evidence)
    print(json.dumps(dict(phase='full144M_pairwise_bandwidth_complete', evidence=median_evidence)), flush=True)
    Z, X, CG, CD, CS = model.create_placeholders(28, 28, 5, 28, 0)
    gs = {k: settings[k] for k in ('hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale')}
    ds = {k: settings[k] for k in ('hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean')}
    dl, gl = model.GAN_loss(Z, X, gs, ds, settings['kappa'], False, CG, CD, CS, wrong_labels=False)
    dsolver, gsolver, accountant = model.GAN_solvers(dl, gl, settings['learning_rate'], 28,
        samples['train'].shape[0], settings['l2norm_bound'], settings['batches_per_lot'], settings['dp_sigma'], dp=False)
    generated = model.generator(Z, **gs, reuse=True)
    model_parameters = int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables()))
    if model_parameters != 96929:
        raise ValueError('Full original96929-parameter model required')
    batch_multiplier = 5000 // 28
    eval_size, fit_size = batch_multiplier * 28, int(.2 * batch_multiplier * 28)
    test_size = eval_size - fit_size
    sigma = tf.get_variable('sigma', shape=2,
        initializer=tf.constant_initializer(value=np.power(bandwidth, np.linspace(-1, 3, num=2))))
    cases = {}
    for tag, count, biased in [('fit', fit_size, True), ('test', test_size, False)]:
        a = tf.placeholder(tf.float32, [count, 28, 28], name=tag + '_real')
        b = tf.placeholder(tf.float32, [count, 28, 28], name=tag + '_generated')
        native_grams = dict(XX=tf.tensordot(a, a, axes=[[1, 2], [1, 2]]),
            XY=tf.tensordot(a, b, axes=[[1, 2], [1, 2]]), YY=tf.tensordot(b, b, axes=[[1, 2], [1, 2]]))
        grams = {key: tf.placeholder(tf.float32, [count, count], name=tag + '_cached_' + key) for key in native_grams}
        kernels = cached_kernel(mmd, a, b, sigma, grams)
        cached = mmd._mmd2_and_ratio(*kernels[:3], const_diagonal=kernels[3], biased=biased)
        original = mmd.mix_rbf_mmd2_and_ratio(a, b, sigma, biased=biased)
        cases[tag] = dict(a=a, b=b, native_grams=native_grams, grams=grams, cached=cached, original=original,
            cached_gradient=tf.gradients(-cached[1], sigma)[0], original_gradient=tf.gradients(-original[1], sigma)[0], kernels=kernels)
    with tf.variable_scope('SIGMA_optimizer'):
        sigma_solver = tf.train.RMSPropOptimizer(learning_rate=.05).minimize(-cases['fit']['cached'][1], var_list=[sigma])
    sigma_opt_vars = [v for v in tf.global_variables() if 'SIGMA_optimizer' in v.name]
    init = tf.global_variables_initializer()
    reset_sigma_slots = tf.variables_initializer(sigma_opt_vars)
    config_tf = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
    output = ROOT / config['artifact_directory']
    output.mkdir(parents=True, exist_ok=False)
    with tf.Session(config=config_tf) as session:
        session.run(init)
        # Keep the author's pre-evaluation visualization RNG consumption/order.
        vis_z = model.sample_Z(28, 28, 5, False)
        session.run(generated, feed_dict={Z: vis_z})
        vis_real_indices = np.random.choice(len(samples['vali']), size=6)
        eval_z = model.sample_Z(eval_size, 28, 5, False)
        eval_sample = np.empty((eval_size, 28, 28))
        for index in range(batch_multiplier):
            eval_sample[index * 28:(index + 1) * 28] = session.run(generated, feed_dict={Z: eval_z[index * 28:(index + 1) * 28]})
        eval_sample = np.float32(eval_sample)
        eval_indices = np.random.choice(len(samples['vali']), size=eval_size)
        eval_real = np.float32(samples['vali'][eval_indices])
        arrays = output / 'full_native_evaluator_inputs.npz'
        np.savez_compressed(str(arrays), real=eval_real, generated=eval_sample, noise=eval_z,
                            validation_indices=eval_indices, visualization_indices=vis_real_indices)
        feeds, comparisons = {}, []
        for tag, selection in [('fit', slice(0, fit_size)), ('test', slice(fit_size, None))]:
            case = cases[tag]
            feed = {case['a']: eval_real[selection], case['b']: eval_sample[selection]}
            values = session.run(case['native_grams'], feed_dict=feed)
            feed.update({case['grams'][key]: value for key, value in values.items()})
            feeds[tag] = feed
            cached_values, original_values, cached_grad, original_grad = session.run(
                [case['cached'], case['original'], case['cached_gradient'], case['original_gradient']], feed_dict=feed)
            for actual, expected in [(cached_values, original_values), (cached_grad, original_grad)]:
                if not np.isfinite(actual).all() or not np.isfinite(expected).all() or not np.allclose(actual, expected,
                        atol=config['numerical_tolerance']['atol'], rtol=config['numerical_tolerance']['rtol']):
                    raise ValueError('Complete original vs cached native MMD/ratio/sigma-gradient differs')
            comparisons.append(dict(phase=tag, full_points=len(eval_real[selection]), native=list(map(float, original_values)),
                cached=list(map(float, cached_values)), maximum_gradient_difference=float(np.max(np.abs(cached_grad - original_grad)))))
        session.run(reset_sigma_slots)
        sigma_iter, that_change, old_that = 0, .002, 0
        while that_change > .001 and sigma_iter < 2000:
            new_sigma, that_np, unused = session.run([sigma, cases['fit']['cached'][1], sigma_solver], feed_dict=feeds['fit'])
            if not np.isfinite(new_sigma).all() or not np.isfinite(that_np):
                raise ValueError('Original full-budget sigma optimizer became nonfinite')
            that_change = float(np.abs(that_np - old_that))
            old_that = that_np
            sigma_iter += 1
            if sigma_iter % 100 == 0:
                print(json.dumps(dict(phase='full_native_sigma_fit', updates=sigma_iter, maximum_updates=2000)), flush=True)
        original_test, cached_test = session.run([cases['test']['original'], cases['test']['cached']], feed_dict=feeds['test'])
        if not np.allclose(original_test, cached_test, **config['numerical_tolerance']) or not np.isfinite(original_test).all():
            raise ValueError('Full optimized original vs cached test MMD differs')
        operations_before_literal = set(tf.get_default_graph().get_operations())
        literal = mmd.mix_rbf_mmd2_and_ratio(eval_real[fit_size:], eval_sample[fit_size:], sigmas=sigma, biased=False)
        literal_values = session.run(literal)
        if not np.allclose(literal_values, cached_test, **config['numerical_tolerance']) or not np.isfinite(literal_values).all():
            raise ValueError('Literal complete original NumPy-input evaluator differs from cached evaluator')
        literal_added_ops = set(tf.get_default_graph().get_operations()) - operations_before_literal
        literal_constant_bytes = sum(len(op.get_attr('value').tensor_content) for op in literal_added_ops if op.type == 'Const')
        # Record the original ratio-function diagonal convention separately.
        kernels = cases['test']['kernels']
        proper_unbiased = mmd._mmd2(*kernels[:3], const_diagonal=kernels[3], biased=False)
        unbiased_comparison = float(session.run(proper_unbiased, feed_dict=feeds['test']))
        fixed_ops = len(tf.get_default_graph().get_operations())
        for unused in range(3):
            session.run(cases['test']['cached'], feed_dict=feeds['test'])
        if len(tf.get_default_graph().get_operations()) != fixed_ops:
            raise ValueError('Cached repeat evaluator must not grow graph')
        final_sigma = session.run(sigma).tolist()
    after = verify_source(asset)
    if before != after or sha(dataset['derived_training_array']) != config['derived_training_sha256']:
        raise ValueError('Original source or data changed during audit')
    return dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), source_commit=asset['commit'],
        all_source_files_verified=len(before), source_and_data_unchanged=True, full_split_shapes=config['full_split_shapes'],
        model_parameters=model_parameters, gan_optimizer_updates=0, generated_full_evaluator_samples=eval_size,
        generator_forward_calls=batch_multiplier, complete_sigma_fit_points=fit_size, complete_test_evaluator_points=test_size,
        bandwidth_audit=median_evidence, full_native_cache_comparisons=comparisons,
        sigma_fit_optimizer_updates=sigma_iter, sigma_fit_maximum_updates=2000, original_sigma_stopping_delta=.001,
        observed_sigma_stopping_delta=that_change, full_original_sigma_stopping_budget_preserved=True, final_sigma=final_sigma,
        optimized_test_mmd_original=float(original_test[0]), optimized_test_ratio_original=float(original_test[1]),
        optimized_test_mmd_cached=float(cached_test[0]), original_ratio_function_diagonal_convention_preserved=True,
        literal_original_full_numpy_input_mmd=float(literal_values[0]), literal_original_full_numpy_input_ratio=float(literal_values[1]),
        original_per_evaluation_added_graph_operations=len(literal_added_ops), original_per_evaluation_added_constant_tensor_bytes=literal_constant_bytes,
        original_mmd2_helper_unbiased_value=unbiased_comparison,
        original_ratio_minus_unbiased_helper=float(original_test[0] - unbiased_comparison),
        graph_operation_count=fixed_ops, repeat_cache_graph_growth=0,
        full_native_evaluator_inputs=dict(path=str(arrays), sha256=sha(arrays)),
        preparation_only_not_benchmark_performance=True, full500_epoch_training_completed=False,
        benchmark_metric_rows_added=0, author_equivalence_certified=False, all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_directory'] / (args.case + '.json')
    if target.exists():
        raise FileExistsError('Preserve every native evaluator attempt')
    try:
        proof = audit(config)
    except Exception as exc:
        import traceback
        write(target, dict(passed=False, error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc(),
                           benchmark_performance=False, all_original_experiments_complete=False))
        raise
    write(target, proof)


if __name__ == '__main__':
    main()
