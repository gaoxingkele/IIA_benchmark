"""Reconstruct every interrupted native epoch's RNG and cumulative data order.

This executes the unchanged author random-draw control flow with an explicit
non-training session. It cannot produce GAN losses or benchmark performance.
Every draw sequence must match saved full evaluator noise/real pixels and every
post-shuffle checkpoint RNG state before its final permutation may be reused.
"""
import argparse
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def extract_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names) or {node.name for node in selected} != set(names):
        raise ValueError('Every exact original random-protocol function is required')
    tree.body = selected
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen interrupted native epoch evidence changed: ' + item['path'])
    queue = read(ROOT / config['queue'])
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration as verify_queue
    verify_queue(queue)
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    output = ROOT / job['output_directory']
    progress = read(output / 'progress.json')
    receipt = read(ROOT / config['failed_terminal_receipt'])
    count = config['completed_original_epochs']
    if (receipt['exit_code'] == 0 or receipt['full_native_result_verified']
            or not receipt['resource_guard_aborted'] or progress['job'] != job
            or count != 238 or progress['full_epochs_completed'] != count
            or progress['required_epochs'] != 500 or progress['full_training_complete']
            or any(progress['optimizer_updates'][k] != count * 1284 for k in ('D', 'G'))):
        raise ValueError('Only the authoritative interrupted full238-of500-epoch attempt may be audited')
    evaluations = read(output / 'completed_full_evaluators.json')
    if (len(evaluations) != count or [e['epoch'] for e in evaluations] != list(range(count))
            or sum(e['sigma_optimizer_updates'] for e in evaluations) != progress['optimizer_updates']['sigma']):
        raise ValueError('All completed native evaluator and update ledgers are required')
    for evaluation in evaluations:
        if (evaluation['complete_generated_points'] != 4984 or evaluation['full_fit_points'] != 996
                or evaluation['full_test_points'] != 3988
                or not 1 <= evaluation['sigma_optimizer_updates'] <= 2000
                or (evaluation['sigma_optimizer_updates'] < 2000 and evaluation['stopping_delta'] > .001)):
            raise ValueError('Preserve every complete original evaluator stopping budget')
    return queue, job, output, evaluations


def compare_state(actual, saved, np):
    expected = (str(saved['algorithm'].item()), saved['keys'], int(saved['position'].item()),
                int(saved['has_gauss'].item()), float(saved['cached_gaussian'].item()))
    if (actual[0] != expected[0] or not np.array_equal(actual[1], expected[1])
            or actual[2:] != expected[2:]):
        raise ValueError('The complete original post-epoch MT19937 state must match bitwise')


def nested_tuple(value):
    return tuple(nested_tuple(item) for item in value) if isinstance(value, list) else value


class RandomProtocolSession:
    """Observe author draw sites only; no TF operations or numerical losses."""
    def __init__(self, np):
        self.np = np
        self.counts = dict(D=0, G=0, loss=0)
        self.loss_noise = None
        self.loss_batch_ids = None

    def run(self, fetches, feed_dict):
        if fetches in ('D', 'G'):
            self.counts[fetches] += 1
            return None
        if fetches == ['D_loss', 'G_loss']:
            self.counts['loss'] += 1
            self.loss_noise = feed_dict['Z'].copy()
            self.loss_batch_ids = feed_dict['X'].copy()
            return self.np.zeros(1), self.np.zeros(1)
        raise ValueError('Unexpected original RNG-only control-flow operation')


def worker(config, config_path):
    queue, job, output, evaluations = verify_registration(config)
    import numpy as np
    if (Path(sys.executable).resolve() != Path(queue['python']).resolve()
            or np.__version__ != '1.18.5' or sys.version != queue['frozen_python_version']):
        raise ValueError('Use the exact original native NumPy and Python environment')
    import pkg_resources
    package_root = (Path(sys.prefix) / 'Lib/site-packages').resolve()
    packages = {p.key:p.version for p in pkg_resources.working_set if Path(p.location).resolve() == package_root}
    if packages != queue['frozen_native_packages'] or 'tensorflow' in sys.modules:
        raise ValueError('Exact frozen native packages and NumPy-only audit required')
    for item in queue['native_interpreter_receipts']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen native interpreter changed')
    registry = read(ROOT / queue['source_registry'])
    asset = next(a for a in registry['assets'] if a['id'] == 'rcgan')
    from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
    original_source = verify_source(asset)
    source = Path(asset['original_directory'])
    settings = read(source / queue['source_settings'])
    expected = dict(batch_size=28, seq_length=28, latent_dim=5, D_rounds=1, G_rounds=1,
                    cond_dim=0, use_time=False, WGAN_clip=False, shuffle=True,
                    num_epochs=500, num_samples=60000, normalise=False,
                    multivariate_mnist=True, predict_labels=False)
    if any(settings[k] != value for k, value in expected.items()):
        raise ValueError('Do not apply the verified MNIST RNG protocol to a different experiment')
    target = ROOT / config['output_directory']
    artifacts = ROOT / config['artifact_root']
    if target.exists() or artifacts.exists():
        raise FileExistsError('Preserve every partial or completed chain audit')
    target.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    dataset = read(ROOT / queue['dataset_configuration'])
    if sha(dataset['derived_training_array']) != queue['derived_training_sha256']:
        raise ValueError('Complete original MNIST data changed')
    table = np.load(dataset['derived_training_array'], mmap_mode='r', allow_pickle=False)
    np.random.seed(job['seed'])
    random.seed(job['seed'])
    initial = np.random.permutation(60000)
    with np.load(str(output / 'full_native_split_indices.npz'), allow_pickle=False) as splits:
        for tag, start, end in [('train', 0, 36000), ('vali', 36000, 48000), ('test', 48000, 60000)]:
            if not np.array_equal(initial[start:end], splits[tag]):
                raise ValueError('Every original full split identity must match the canonical seed')
    data_namespace = dict(np=np)
    extract_functions(source / 'data_utils.py', {'get_batch'}, data_namespace)
    namespace = dict(np=np, data_utils=SimpleNamespace(get_batch=data_namespace['get_batch']))
    extract_functions(source / 'model.py', {'sample_Z', 'sample_C', 'train_epoch'}, namespace)
    # Source initial visualization draws: unconditional fixed Z then six real indices.
    vis_noise = namespace['sample_Z'](28, 28, 5, False)
    vis_ids = initial[36000:48000][np.random.choice(12000, size=6)]
    order = np.arange(36000)
    train_ids = initial[:36000].copy()
    train_labels = np.asarray(table[train_ids, 0])
    session = RandomProtocolSession(np)
    train_keys = ('batch_size', 'use_time', 'D_rounds', 'G_rounds', 'seq_length', 'latent_dim',
                  'num_generated_features', 'cond_dim', 'max_val', 'WGAN_clip', 'one_hot')
    records, loss_noises, loss_ids, choices = [], [], [], []
    for epoch, evaluation in enumerate(evaluations):
        if sha(evaluation['array_path']) != evaluation['array_sha256']:
            raise ValueError('Saved complete evaluator changed')
        namespace['train_epoch'](epoch, train_ids, train_labels, session, 'Z', 'X', 'CG', 'CD', 'CS',
                                 'D_loss', 'G_loss', 'D', 'G', **{k:settings[k] for k in train_keys})
        if session.counts != dict(D=(epoch+1)*1284, G=(epoch+1)*1284, loss=epoch+1):
            raise ValueError('The actual source draw sites must cover all1284D/1284G plus loss draws')
        noise = namespace['sample_Z'](4984, 28, 5, False)
        if namespace['sample_C'](4984, 0, settings['max_val'], settings['one_hot']) is not None:
            raise ValueError('The native unconditional branch must have no label draws')
        real_ids = initial[36000:48000][np.random.choice(12000, size=4984)]
        with np.load(evaluation['array_path'], allow_pickle=False) as saved:
            if noise.shape != (4984, 28, 5) or not np.array_equal(noise, saved['noise']):
                raise ValueError('Every complete original evaluation noise draw must match bitwise')
            pixels = np.float32(table[real_ids, 1:].reshape(4984, 28, 28))
            if not np.array_equal(pixels, saved['real']):
                raise ValueError('Every complete original sampled real image must match bitwise')
        loss_noises.append(session.loss_noise)
        loss_ids.append(session.loss_batch_ids)
        choices.append(real_ids)
        permutation = np.random.permutation(36000)
        order, train_ids, train_labels = order[permutation], train_ids[permutation], train_labels[permutation]
        directory = output / 'native_checkpoints' / ('epoch_%03d' % epoch)
        state_path, python_path = directory / 'numpy_random_state.npz', directory / 'python_random_state.json'
        with np.load(str(state_path), allow_pickle=False) as saved_state:
            compare_state(np.random.get_state(), saved_state, np)
        if random.getstate() != nested_tuple(read(python_path)):
            raise ValueError('Complete Python random state must also match every original checkpoint')
        records.append(dict(epoch=epoch, full4984_noise_and_real_pixels_bitwise_verified=True,
            post_shuffle_numpy_and_python_states_bitwise_verified=True,
            evaluation_sha256=evaluation['array_sha256'], numpy_state_sha256=sha(state_path),
            python_state_sha256=sha(python_path)))
        write(target / 'progress.json', dict(complete_verified_epochs=len(records), required_epochs=238,
            native_tensorflow_training_executed=False, benchmark_performance=False))
        if epoch % 10 == 0:
            print('verified_full_random_order_epoch=%d' % epoch, flush=True)
    order_path = artifacts / 'verified_random_order_chain.npz'
    np.savez_compressed(str(order_path), cumulative_train_order=order, cumulative_train_image_ids=train_ids,
        initial_split_permutation=initial, initial_visualization_noise=vis_noise, initial_visualization_image_ids=vis_ids,
        original_epoch_loss_noise=np.stack(loss_noises), original_epoch_loss_batch_image_ids=np.stack(loss_ids),
        original_full_evaluator_image_ids=np.stack(choices))
    if not np.array_equal(np.sort(order), np.arange(36000)) or not np.array_equal(initial[:36000][order], train_ids):
        raise ValueError('Cumulative training order must preserve all original36000 image identities')
    if original_source != verify_source(asset):
        raise ValueError('Original author source changed during chain audit')
    verify_registration(config)
    write(target / 'random_order_chain_audit.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=sha(ROOT / config_path), job=job, original_completed_epochs=238,
        required_original_epochs=500, verified_original_numpy_version=np.__version__,
        complete_full_rng_and_cumulative_data_order_verified=True, verified_epoch_records=records,
        protocol_draw_counts=session.counts, original_full_evaluator_points_per_epoch=4984,
        derived_order_artifact=dict(path=str(order_path), sha256=sha(order_path)),
        tensorflow_model_updates_executed_by_this_audit=0, numerical_training_losses_reconstructed=False,
        full500_epoch_result_verified=False, benchmark_performance=False, additional_training_seed_slots=0,
        original_scope=config['original_scope'], source_receipts=config['source_receipts'],
        author_equivalence_certified=False, all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify_registration(config)
    if args.probe_only:
        print(dict(complete_original_epochs=238, required_epochs=500,
                   numpy_imported='numpy' in sys.modules, tensorflow_imported='tensorflow' in sys.modules))
        return
    if args.worker:
        worker(config, args.config)
        return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('This chain audit already has a live controller')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.audit_rcgan_random_order_chain_v1', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
                             PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                identity = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=identity.create_time(), actual_command=identity.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT / config['output_directory'] / 'random_order_chain_audit.json'
        passed = bool(code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed'])
        write(base / 'receipt.json', dict(exit_code=code, random_order_chain_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, random_order_chain_passed=passed))
        if not passed:
            raise RuntimeError('Preserve the failed original238epoch chain audit')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
