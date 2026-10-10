"""Audit all full original TimeGAN graphs/data; no shortened training or performance claim."""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, full_budget, verify_source, constructor_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    model = read(ROOT / config['model_config'])
    full_budget(model)
    source, manifest = verify_source(model)
    output = ROOT / config['output_directory']
    if output.exists():
        raise FileExistsError('Preserve native constructor audit attempts')
    output.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import tensorflow as tf
    import timegan as author
    import data_loading as loader
    from metrics import discriminative_metrics as disc, predictive_metrics as pred
    if tf.__version__ != '1.15.0' or not hasattr(tf, 'contrib'):
        raise ValueError('Actual native TF1 contrib environment required')
    data_config = read(ROOT / model['dataset_config'])
    graph = constructor_prefix(author, 'timegan')
    metric_graphs = [constructor_prefix(disc, 'discriminative_score_metrics'),
                     constructor_prefix(pred, 'predictive_score_metrics')]
    session_config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2,
                                   inter_op_parallelism_threads=1)
    records, datasets, metrics = [], [], []
    for name in ('stock', 'energy', 'sine'):
        item = data_config['datasets'][name]
        np.random.seed(1103)
        if name == 'sine':
            data = loader.sine_data_generation(item['samples'], 24, item['features'])
            independently = []
            rng = np.random.RandomState(1103)
            for _ in range(item['samples']):
                cols = []
                for _ in range(item['features']):
                    freq, phase = rng.uniform(0, .1), rng.uniform(0, .1)
                    cols.append([np.sin(freq * j + phase) for j in range(24)])
                independently.append((np.asarray(cols).T + 1) * .5)
            raw_shape, raw_hash = None, None
        else:
            raw_path = ROOT / item['raw_file']
            raw = np.loadtxt(str(raw_path), delimiter=',', skiprows=1)
            raw_hash, raw_shape = sha(raw_path), list(raw.shape)
            if raw_hash != item['raw_sha256']:
                raise ValueError('Original raw data identity differs')
            data = loader.real_data_loading(name, 24)
            reversed_data = raw[::-1]
            norm = (reversed_data - reversed_data.min(0)) / (reversed_data.max(0) - reversed_data.min(0) + 1e-7)
            windows = np.asarray([norm[i:i+24] for i in range(len(raw)-24)])
            independently = windows[np.random.RandomState(1103).permutation(len(windows))]
        array = np.asarray(data)
        if list(array.shape) != item['full_shape'] or not np.array_equal(array, independently) or not np.isfinite(array).all():
            raise ValueError('Complete native loader differs from independent full-data calculation')
        datasets.append(dict(dataset=name, raw_shape=raw_shape, raw_sha256=raw_hash,
                             full_shape=list(array.shape), array_sha256=__import__('hashlib').sha256(array.tobytes()).hexdigest(),
                             independent_full_loader_bitwise_identical=True,
                             boundary='All windows used for native generation training; posthoc random overlapping-window splits are not grouped TAB evaluation.'))
        for cell in ('gru', 'lstm', 'lstmLN'):
            p = dict(module=cell, hidden_dim=24, num_layer=3, iterations=50000, batch_size=128)
            nodes = graph(data, p)
            variables = tf.trainable_variables()
            sizes = [dict(name=v.name, shape=v.shape.as_list()) for v in variables]
            params = int(sum(np.prod(v.shape.as_list()) for v in variables))
            with tf.Session(config=session_config) as sess:
                sess.run(tf.global_variables_initializer())
                minibatch = nodes['ori_data'][:128]
                noise = np.random.RandomState(1103).uniform(0, 1, (128, 24, array.shape[2]))
                generated, *losses = sess.run([nodes['X_hat'], nodes['E_loss0'], nodes['G_loss'], nodes['D_loss']],
                                              feed_dict={nodes['X']: minibatch, nodes['Z']: noise, nodes['T']: [24]*128})
            if generated.shape != (128, 24, array.shape[2]) or not np.isfinite(generated).all() or not np.isfinite(losses).all():
                raise ValueError('Actual original full native network forward failed')
            records.append(dict(dataset=name, module=cell, full_native_batch=128, model_parameters=params,
                                full_native_hidden_dim=24, full_native_layers=3, variables=sizes,
                                output_shape=list(generated.shape), finite_forward=True, optimizer_updates=0))
            write(output / 'progress.json', dict(graphs=len(records), latest=records[-1]))
        for kind, fn in zip(('discriminative', 'predictive'), metric_graphs):
            nodes = fn(data, data)
            metrics.append(dict(dataset=name, metric=kind, parameters=int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables())),
                                hidden_dim=nodes['hidden_dim'], native_updates=nodes['iterations'], native_batch=nodes['batch_size']))
        del data, array, independently
        gc.collect()
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), config=args.config, config_sha256=sha(ROOT / args.config),
                  environment=dict(python=sys.version, tensorflow=tf.__version__, numpy=np.__version__),
                  native_tf_contrib_available=True, full_original_graphs_passed=9,
                  full_original_metric_graphs_passed=6, datasets=datasets, records=records, metric_graphs=metrics,
                  native_training_budget=full_budget(model), source_files=manifest['files'],
                  benchmark_performance=False, full_training_completed=False, all_original_experiments_complete=False,
                  author_equivalence_certified=False,
                  boundary='Full original native network construction, batch128 forward and independent complete original loader audit only. Actual training must execute untouched original function with all50000+50000+50000 stages and full10x2000/10x5000 evaluator trainings. Defaults and environment locally frozen; historical optimized author settings unknown; adopted irregular comparator tasks remain. Synthetic sine is an original published task, but this constructor audit is not its performance.')
    write(output / 'full_native_graph_and_data_audit.json', report)
    print('Full original graphs9, metric graphs6, datasets3 verified', flush=True)


if __name__ == '__main__':
    main()
