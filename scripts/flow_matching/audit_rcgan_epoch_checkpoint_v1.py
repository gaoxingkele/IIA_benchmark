"""Replay every generated evaluator sample from a complete native epoch checkpoint."""
import argparse
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration
    verify_registration(queue)
    if Path(sys.executable).resolve() != Path(queue['python']).resolve() or not 0 <= args.epoch < queue['epochs']:
        raise ValueError('Designated full native experiment required')
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    output = ROOT / job['output_directory']
    progress = read(output / 'progress.json')
    if progress['full_epochs_completed'] <= args.epoch:
        raise ValueError('An actually completed original full epoch is required')
    if any(progress['optimizer_updates'][k] != progress['full_epochs_completed'] * queue[k + '_updates_per_epoch']
           for k in ['D', 'G']):
        raise ValueError('Complete actual full update ledger required')
    import numpy as np
    import tensorflow as tf
    directory = output / 'native_checkpoints' / ('epoch_%03d' % args.epoch)
    checkpoint = str(directory / 'model')
    evaluation = next(e for e in read(output / 'completed_full_evaluators.json') if e['epoch'] == args.epoch)
    if sha(evaluation['array_path']) != evaluation['array_sha256']:
        raise ValueError('Full evaluator archive changed')
    with np.load(evaluation['array_path'], allow_pickle=False) as archive:
        real, generated, noise, sigma = [archive[k] for k in ['real', 'generated', 'noise', 'sigma']]
    if real.shape != (4984, 28, 28) or generated.shape != real.shape or noise.shape != (4984, 28, 5):
        raise ValueError('All original4984 evaluator samples and source noise required')
    if not all(np.isfinite(v).all() for v in [real, generated, noise, sigma]):
        raise ValueError('Nonfinite full evaluator values')
    original_dump = output / 'experiments/parameters' / (job['id'] + '_' + str(args.epoch) + '.npy')
    parameters = np.load(str(original_dump), allow_pickle=True).item()
    reader = tf.train.load_checkpoint(checkpoint)
    variables = reader.get_variable_to_shape_map()
    if not any('/Adam' in name for name in variables) or not any('/RMSProp' in name for name in variables):
        raise ValueError('Original generator Adam and sigma RMSProp states required')
    if not all(np.isfinite(reader.get_tensor(name)).all() for name in variables):
        raise ValueError('Nonfinite full native checkpoint state')
    for name, value in parameters.items():
        if not name.endswith(':0') or not np.array_equal(reader.get_tensor(name[:-2]), value):
            raise ValueError('Every original dumped trainable parameter must match the full checkpoint bitwise')
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
    trainable = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if {v.name for v in trainable} != set(parameters):
        raise ValueError('Full original trainable-variable membership required')
    gan_parameters = sum(int(np.prod(v.shape.as_list())) for v in trainable
                         if v.name.startswith(('generator/', 'discriminator/')))
    if gan_parameters != 96929:
        raise ValueError('Complete original96929-parameter GAN required')
    candidates = [t for op in graph.get_operations() for t in op.outputs
                  if op.name.startswith('generator/') and op.type == 'Reshape'
                  and t.shape.as_list() == [28, 28, 28]]
    inputs = [t for op in graph.get_operations() if op.type == 'Placeholder'
              for t in op.outputs if t.shape.as_list() == [28, 28, 5]]
    if len(candidates) != 1 or len(inputs) != 1:
        raise ValueError('Unique complete original generator output and noise input required')
    replay = np.empty_like(generated)
    with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU': 0},
                    intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)) as session:
        saver.restore(session, checkpoint)
        for start in range(0, 4984, 28):
            replay[start:start + 28] = session.run(candidates[0], feed_dict={inputs[0]: noise[start:start + 28]})
    if not np.array_equal(replay, generated):
        raise ValueError('Every original evaluator generator output must replay bitwise')
    for name in ['numpy_random_state.npz', 'python_random_state.json']:
        if not (directory / name).exists():
            raise ValueError('Complete original random-state checkpoint required')
    artifacts = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)
                 for p in sorted(directory.iterdir()) if p.is_file()]
    write(ROOT / args.report, dict(passed=True, job=job, audited_epoch=args.epoch,
        actual_full_progress=progress, native_python=sys.executable, full_native_GAN_parameters=gan_parameters,
        native_checkpoint_variable_count=len(variables), original_trainable_variable_count=len(parameters),
        all_original_dumped_parameters_bitwise_equal_full_checkpoint=True,
        generator_Adam_and_sigma_RMSProp_state_present=True, all_checkpoint_variables_finite=True,
        all4984_generator_samples_replayed_bitwise=True, original_full_generator_calls=178,
        max_absolute_generator_replay_difference=float(np.max(np.abs(replay - generated))),
        original_generator_tensor=candidates[0].name, original_noise_tensor=inputs[0].name,
        full_native_evaluator=evaluation, original_parameter_dump_path=str(original_dump),
        original_parameter_dump_sha256=sha(original_dump), checkpoint_artifacts=artifacts,
        no_additional_training_seed_created=True, full500_epoch_result_verified=False,
        benchmark_performance=False, author_equivalence_certified=False, all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
