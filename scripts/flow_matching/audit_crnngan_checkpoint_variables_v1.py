"""Independently inspect every tensor in a real native C-RNN-GAN checkpoint."""
import argparse
import hashlib
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def verify_tensors(reader, audit):
    import numpy as np
    expected = {item['name']: item for item in audit['tensors']}
    if len(expected) != audit['full_variable_count'] or set(expected) != set(reader.get_variable_to_shape_map()):
        raise ValueError('Every full saved native tensor must be independently read')
    parameters = 0
    for name, item in expected.items():
        array = reader.get_tensor(name)
        if (list(array.shape) != item['shape'] or str(array.dtype) != item['dtype']
                or not np.isfinite(array).all()
                or hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != item['sha256']):
            raise ValueError('Full saved native tensor differs: ' + name)
        if array.shape:
            parameters += int(array.size)
    return parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    if Path(sys.executable).resolve() != Path(queue['python']).resolve():
        raise ValueError('Frozen native checkpoint interpreter required')
    import numpy as np
    import tensorflow as tf
    if tf.__version__ != '1.15.0' or np.__version__ != '1.18.5':
        raise ValueError('Frozen native checkpoint numerical packages required')
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    directory = ROOT / job['output_directory'] / 'native_checkpoints' / ('epoch_%03d' % args.epoch)
    audit = read(directory / 'full_checkpoint_variable_audit.json')
    if sha(audit['meta_graph']) != audit['meta_graph_sha256']:
        raise ValueError('Full native metagraph changed')
    reader = tf.train.NewCheckpointReader(str(directory / 'model'))
    parameters = verify_tensors(reader, audit)
    progress = read(ROOT / job['output_directory'] / 'progress.json')
    epoch = progress['epochs'][args.epoch]
    for artifact in epoch['checkpoint_artifacts']:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Complete original checkpoint or RNG artifact changed')
    if parameters != progress['model_constructions'][0]['trainable_parameters']:
        raise ValueError('Full original model width/depth parameter count changed')
    write(ROOT / args.output, dict(job=job, epoch=args.epoch, full_saved_native_tensors=audit['full_variable_count'],
        full_original_trainable_parameters=parameters, all_saved_tensor_hashes_equal_live_training_receipt=True,
        all_original_checkpoint_and_RNG_file_hashes_verified=True, optimizer_updates=epoch['optimizer_updates'],
        full_training_batches=epoch['full_training_batches'], full_original_epoch=epoch,
        checkpoint_variable_audit_sha256=sha(directory / 'full_checkpoint_variable_audit.json'),
        actual_python=sys.executable, tensorflow=tf.__version__, numpy=np.__version__,
        model_prediction_replay_certified=False, full500_training_completed=False, benchmark_performance=False,
        all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
