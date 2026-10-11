"""Observe the full author MNIST experiment with audited evaluator bridges.

All GAN source functions and the complete500-epoch source loop remain intact.
The unused eval import is omitted after proving it has no references; no
algorithm is stubbed. Raw author source and inputs remain read-only.
"""
import ast
import inspect
import os
from pathlib import Path
import random
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source


def experiment_ast(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    if any(isinstance(node, ast.Name) and node.id == 'eval' for node in ast.walk(tree)):
        raise ValueError('Do not omit a referenced original evaluator')
    removed = 0
    for node in list(tree.body):
        if isinstance(node, ast.Import) and [item.name for item in node.names] == ['eval']:
            tree.body.remove(node)
            removed += 1
    loops = [node for node in tree.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
             and node.target.id == 'epoch']
    if removed != 1 or len(loops) != 1:
        raise ValueError('Exactly one unused import and one full original epoch loop required')
    loops[0].body.extend(ast.parse('_native_epoch_complete(epoch, locals())').body)
    return ast.fix_missing_locations(tree)


def run(queue, job):
    import numpy as np
    import tensorflow as tf
    from scripts.flow_matching.run_timegan_native_job_v1 import flatten_fetches
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full training registration changed')
    proof = read(ROOT / queue['native_evaluator_proof'])
    if sha(ROOT / queue['native_evaluator_proof']) != queue['native_evaluator_proof_sha256'] or not proof['passed']:
        raise ValueError('Full original native evaluator proof required before training')
    if (proof['model_parameters'] != 96929 or proof['generated_full_evaluator_samples'] != 4984
            or proof['complete_sigma_fit_points'] != 996 or proof['complete_test_evaluator_points'] != 3988
            or not proof['full_original_sigma_stopping_budget_preserved']
            or not proof['sigma_fit_uses_unchanged_original_function_and_gradient']
            or not proof['production_training_uses_no_gram_caching']
            or not proof['reusable_original_test_input_graph_compared_to_literal_author_numpy_call']):
        raise ValueError('Full original model/evaluator budget required')
    registry = read(ROOT / queue['source_registry'])
    asset = next(a for a in registry['assets'] if a['id'] == 'rcgan')
    before = verify_source(asset)
    if Path(sys.executable).resolve() != Path(queue['python']).resolve() or tf.__version__ != '1.15.0' or np.__version__ != '1.18.5':
        raise ValueError('Frozen native interpreter/numerical packages required')
    import pkg_resources
    package_root = (Path(sys.prefix) / 'Lib/site-packages').resolve()
    actual_packages = {p.key: p.version for p in pkg_resources.working_set if Path(p.location).resolve() == package_root}
    if actual_packages != queue['frozen_native_packages'] or sys.version != queue['frozen_python_version']:
        raise ValueError('Complete frozen isolated native numerical environment required')
    for item in queue['native_interpreter_receipts']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Native interpreter binary or path configuration changed')
    output = ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every full original training attempt')
    output.mkdir(parents=True)
    for name in ['data', 'experiments/settings', 'experiments/data', 'experiments/parameters', 'experiments/traces', 'experiments/plots']:
        (output / name).mkdir(parents=True, exist_ok=True)
    dataset = read(ROOT / queue['dataset_configuration'])
    if sha(dataset['derived_training_array']) != queue['derived_training_sha256']:
        raise ValueError('Original complete training input changed')
    os.link(dataset['derived_training_array'], str(output / 'data/mnist_train.npy'))
    settings = read(Path(asset['original_directory']) / queue['source_settings'])
    settings['identifier'] = job['id']
    write(output / 'experiments/settings' / (job['id'] + '.txt'), settings)
    os.chdir(str(output))
    sys.path.insert(0, asset['original_directory'])
    os.environ['MPLBACKEND'] = 'Agg'
    import mmd
    import model
    import data_utils
    import plotting
    from scripts.flow_matching.rcgan_plot_trace_compat_v1 import install
    plot_receipt = install(plotting)
    write(output / 'plot_trace_compatibility_receipt.json', plot_receipt)
    np.random.seed(job['seed'])
    random.seed(job['seed'])
    tf.reset_default_graph()
    tf.set_random_seed(job['seed'])
    session_ref = {}
    cache = {}
    epochs, evaluators, checkpoints = [], [], []
    original_loader = data_utils.get_samples_and_labels

    def full_loader(settings):
        permutations = []
        original_permutation = np.random.permutation

        def permutation(value):
            result = original_permutation(value)
            if isinstance(value, (int, np.integer)) and value == 60000:
                permutations.append(result.copy())
            return result

        np.random.permutation = permutation
        try:
            samples, pdf, labels = original_loader(settings)
        finally:
            np.random.permutation = original_permutation
        if len(permutations) != 1 or not np.array_equal(np.sort(permutations[0]), np.arange(60000)):
            raise ValueError('All60000 source image identities must occur exactly once')
        if {key: list(value.shape) for key, value in samples.items()} != queue['full_split_shapes']:
            raise ValueError('Full original36000/12000/12000 native image splits required')
        table = np.load(dataset['derived_training_array'], mmap_mode='r', allow_pickle=False)
        start, identities = 0, {}
        for key in ('train', 'vali', 'test'):
            count = len(samples[key])
            indices = permutations[0][start:start + count]
            for offset in range(0, count, 1000):
                selected = indices[offset:offset + 1000]
                if (not np.array_equal(samples[key][offset:offset + 1000].reshape(-1, 784), table[selected, 1:])
                        or not np.array_equal(labels[key][offset:offset + 1000], table[selected, 0])):
                    raise ValueError('Every native split pixel/label must equal its source image')
            identities[key] = indices
            start += count
        path = output / 'full_native_split_indices.npz'
        np.savez_compressed(str(path), **identities)
        write(output / 'full_native_loader_audit.json', dict(full_original_training_images=60000,
            full_original_split_shapes=queue['full_split_shapes'], grouped_disjoint_image_indices=True,
            every_original_split_pixel_and_label_bitwise_verified=True, full_split_indices_path=str(path),
            full_split_indices_sha256=sha(path), official10000_test_images_used_for_this_author_split=False))
        return samples, pdf, labels

    data_utils.get_samples_and_labels = full_loader
    original_median, original_mixture = mmd.median_pairwise_distance, mmd.mix_rbf_mmd2_and_ratio

    def bandwidth(X, Y=None):
        value = float(original_median(X, Y))
        if len(X) != 12000 or not np.isfinite(value) or value <= 0:
            raise ValueError('Complete original12000-validation bandwidth required')
        write(output / 'full_bandwidth_bridge_receipt.json', dict(full_original_pairs=144000000, value=value,
            original_median_function_used_without_numeric_changes=True))
        return value

    def build_case(tag, count, sigma, biased):
        a = tf.placeholder(tf.float32, [count, 28, 28], name=tag + '_full_real')
        b = tf.placeholder(tf.float32, [count, 28, 28], name=tag + '_full_generated')
        value = original_mixture(a, b, sigma, biased=biased)
        return dict(a=a, b=b, value=value)

    def mixture(X, Y, sigmas=(1,), wts=None, biased=True):
        if wts is not None:
            raise ValueError('Original default full two-kernel weighting required')
        if isinstance(X, tf.Tensor):
            if list(X.shape) != [996, 28, 28] or cache:
                raise ValueError('Exactly one full original sigma fit graph required')
            fit_value = original_mixture(X, Y, sigmas, wts=wts, biased=biased)
            cache['test'] = build_case('test', 3988, sigmas, False)
            cache['sigma'] = sigmas
            # The author fit graph and its derivative are entirely unchanged.
            return fit_value
        if np.asarray(X).shape != (3988, 28, 28) or np.asarray(Y).shape != (3988, 28, 28) or biased or sigmas is not cache['sigma']:
            raise ValueError('Complete original3988-point test MMD recipe required')
        native_locals = inspect.currentframe().f_back.f_locals
        session = session_ref['session']
        iterations = int(native_locals['sigma_iter'])
        if session.counts['sigma'] - session.last_sigma_count != iterations or not 1 <= iterations <= 2000:
            raise ValueError('All actually executed native sigma steps required')
        if iterations < 2000 and float(native_locals['that_change']) > .001:
            raise ValueError('Original delta stopping rule must remain unchanged')
        session.last_sigma_count = session.counts['sigma']
        cache['pending_test'] = (X, Y)
        sigma = session.run(cache['sigma'])
        path = output / 'full_evaluations' / ('epoch_%03d.npz' % int(native_locals['epoch']))
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(str(path), real=native_locals['eval_real'], generated=native_locals['eval_sample'],
                            noise=native_locals['eval_Z'], sigma=sigma)
        evaluators.append(dict(epoch=int(native_locals['epoch']), sigma_optimizer_updates=iterations,
            complete_generated_points=4984, full_fit_points=996, full_test_points=3988,
            array_path=str(path), array_sha256=sha(path), stopping_delta=float(native_locals['that_change'])))
        return cache['test']['value']

    native_session = tf.Session

    class ObservedSession(native_session):
        def __init__(self, *args, **kwargs):
            caller = inspect.currentframe().f_back.f_locals
            self.ops = {key: caller[value] for key, value in [('D', 'D_solver'), ('G', 'G_solver'), ('sigma', 'sigma_solver')]}
            self.counts = dict(D=0, G=0, sigma=0)
            self.last_sigma_count = 0
            kwargs['config'] = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
            super(ObservedSession, self).__init__(*args, **kwargs)
            session_ref['session'] = self

        def run(self, fetches, *args, **kwargs):
            feed = dict(kwargs.get('feed_dict') or {})
            if 'test' in cache and fetches is cache['test']['value']:
                a, b = cache['pending_test']
                feed.update({cache['test']['a']: a, cache['test']['b']: b})
            if feed:
                kwargs['feed_dict'] = feed
            result = super(ObservedSession, self).run(fetches, *args, **kwargs)
            for name, op in self.ops.items():
                if any(value is op for value in flatten_fetches(fetches)):
                    self.counts[name] += 1
            if 'test' in cache and fetches is cache['test']['value']:
                if not np.isfinite(result).all():
                    raise ValueError('Every full native epoch evaluation must be finite')
                evaluators[-1].update(native_mmd2=float(result[0]), native_ratio=float(result[1]))
            return result

    def epoch_complete(epoch, local):
        session = session_ref['session']
        if epoch != len(epochs) or session.counts['D'] != (epoch + 1) * 1284 or session.counts['G'] != (epoch + 1) * 1284:
            raise ValueError('Exact original1284D/1284G native updates per full epoch required')
        if len(evaluators) != epoch + 1 or evaluators[-1]['epoch'] != epoch:
            raise ValueError('Every original full epoch evaluator required')
        if not np.isfinite([local['D_loss_curr'], local['G_loss_curr']]).all():
            raise ValueError('Original full training losses became nonfinite')
        directory = output / 'native_checkpoints' / ('epoch_%03d' % epoch)
        directory.mkdir(parents=True)
        if not hasattr(session, 'native_saver'):
            session.native_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
        session.native_saver.save(session, str(directory / 'model'))
        state = np.random.get_state()
        np.savez_compressed(str(directory / 'numpy_random_state.npz'), algorithm=state[0], keys=state[1],
                            position=state[2], has_gauss=state[3], cached_gaussian=state[4])
        write(directory / 'python_random_state.json', random.getstate())
        artifacts = [dict(path=str(p), sha256=sha(p)) for p in directory.iterdir() if p.is_file()]
        checkpoints.append(dict(epoch=epoch, optimizer_updates=dict(session.counts), artifacts=artifacts))
        epochs.append(dict(epoch=epoch, D_loss=float(local['D_loss_curr']), G_loss=float(local['G_loss_curr'])))
        write(output / 'progress.json', dict(job=job, full_epochs_completed=len(epochs), required_epochs=500,
            optimizer_updates=session.counts, full_training_complete=False, benchmark_performance=False))
        write(output / 'completed_full_evaluators.json', evaluators)

    mmd.median_pairwise_distance = bandwidth
    mmd.mix_rbf_mmd2_and_ratio = mixture
    tf.Session = ObservedSession
    source = Path(asset['original_directory']) / 'experiment.py'
    namespace = dict(__name__='__main__', __file__=str(source), _native_epoch_complete=epoch_complete)
    sys.argv = ['experiment.py', '--settings_file', job['id']]
    exec(compile(experiment_ast(source), str(source), 'exec'), namespace)
    session = session_ref['session']
    if len(epochs) != 500 or len(evaluators) != 500 or len(checkpoints) != 500 or session.counts['D'] != 642000 or session.counts['G'] != 642000:
        raise ValueError('All500 full native epochs/evaluators/checkpoints required')
    after = verify_source(asset)
    if before != after or sha(dataset['derived_training_array']) != queue['derived_training_sha256']:
        raise ValueError('Original source/data changed')
    write(output / 'result.json', dict(job=job, full_original500_epoch_loop_completed=True,
        full_training_completed=True, full_original_evaluators_completed=True, epochs=epochs, evaluators=evaluators,
        checkpoints=checkpoints, optimizer_updates=session.counts, full_original_split_shapes=queue['full_split_shapes'],
        native_evaluator_proof_sha256=queue['native_evaluator_proof_sha256'], source_algorithm_files_unchanged=True,
        full_native_loader_audit=read(output / 'full_native_loader_audit.json'),
        plotting_compatibility_receipt=plot_receipt,
        original_ratio_diagonal_convention_preserved=True, unused_eval_import_omitted=True,
        verified_zero_distance_roundoff_bridge_applied=False, cached_exact_native_gram_contractions=False,
        original_sigma_fit_function_and_gradient_unchanged=True, full_native_test_input_placeholder_graph_reused=True,
        independent_training_seed_slots=1, author_equivalence_certified=False, strict_TAB_result=False,
        original_paper_all_experiments_complete=False, all_original_experiments_complete=False))
    session.close()
