"""Audit original adopted GAN graphs and every released MIDI file, without training.

Run with the registered isolated legacy interpreter. Original sources and raw
data are read-only. A MIDI package-name alias is explicit; algorithm code is
never rewritten. These checks are preparation, not benchmark performance.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback


def read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def git_blob(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()


def verify_source(asset):
    root = Path(asset['original_directory'])
    records = []
    for entry in asset['expected_git_entries']:
        payload = (root / entry['path']).read_bytes()
        if len(payload) != entry['size'] or git_blob(payload) != entry['sha']:
            raise ValueError('Pinned original Git file differs: ' + entry['path'])
        records.append(dict(path=entry['path'], bytes=len(payload), git_blob_sha1=entry['sha'],
                            sha256=hashlib.sha256(payload).hexdigest()))
    return records


def install_midi_alias():
    actual = importlib.import_module('python3_midi')
    previous = sys.modules.get('midi')
    if previous is not None and previous is not actual:
        raise ValueError('Do not replace a different loaded MIDI implementation')
    sys.modules['midi'] = actual
    if not all(hasattr(actual, key) for key in
               ('read_midifile', 'write_midifile', 'NoteOnEvent', 'NoteOffEvent', 'Pattern', 'Track', 'events')):
        raise ValueError('Original MIDI API is unavailable')
    return actual


def midi_audit(asset, case, output):
    install_midi_alias()
    sys.path.insert(0, asset['original_directory'])
    import music_data_utils as author
    import numpy as np
    # Avoid __init__: it downloads unregistered files. Invoke the original
    # read_one_file method directly for EVERY pinned released MIDI file.
    loader = author.MusicDataLoader.__new__(author.MusicDataLoader)
    loader.output_ticks_per_quarter_note = 384.0
    paths = [entry['path'] for entry in asset['expected_git_entries'] if entry['path'].lower().endswith('.mid')]
    if len(paths) != case['required_released_midi_files']:
        raise ValueError('Every released MIDI file must be visited')
    records = []
    for relative in paths:
        path = Path(asset['original_directory']) / relative
        record = dict(path=relative, bytes=path.stat().st_size)
        try:
            song = loader.read_one_file(str(path.parent), path.name, False)
            if song is None:
                raise ValueError('Original reader rejected this file')
            values = np.asarray(song, dtype=np.float64)
            if values.size and (values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all()):
                raise ValueError('All original parsed note events must have four finite features')
            record.update(parsed=True, note_events=len(song), empty=not bool(song))
        except Exception as exc:
            record.update(parsed=False, error_type=type(exc).__name__, error=str(exc))
        records.append(record)
        if len(records) % 250 == 0:
            print(json.dumps(dict(phase='all_released_midi_parsing', visited=len(records), required=len(paths))), flush=True)
    return dict(all_released_files_visited=len(records) == len(paths), records=records,
                files_visited=len(records), files_parsed=sum(r['parsed'] for r in records),
                files_rejected=sum(not r['parsed'] for r in records),
                nonempty_files=sum(r.get('parsed', False) and not r.get('empty', True) for r in records),
                total_parsed_note_events=sum(r.get('note_events', 0) for r in records),
                paper_reported_files=case['paper_reported_midi_files'],
                count_shortfall=case['paper_reported_midi_files'] - len(records),
                paper_dataset_identity_certified=False,
                source_method='MusicDataLoader.read_one_file, unchanged; resolution 384; pace_events=False')


def graph_audit(asset, case, output):
    import numpy as np
    import tensorflow as tf
    np.random.seed(case['seed'])
    tf.reset_default_graph()
    tf.set_random_seed(case['seed'])
    sys.path.insert(0, asset['original_directory'])
    if case['kind'] == 'rcgan_graph':
        import model as author
        import utils
        import data_utils
        settings = vars(utils.rgan_options_parser().parse_args([]))
        settings.update(case.get('overrides', {}))
        samples, pdf, labels = data_utils.get_samples_and_labels(settings)
        expected = case['full_split_shapes']
        if {key: list(value.shape) for key, value in samples.items()} != expected:
            raise ValueError('Full unchanged original data generation/splits required')
        if not all(np.isfinite(value).all() for value in samples.values()):
            raise ValueError('Full generated data must be finite')
        Z, X, CG, CD, CS = author.create_placeholders(settings['batch_size'], settings['seq_length'],
                                 settings['latent_dim'], settings['num_signals'], settings['cond_dim'])
        gs = {key: settings[key] for key in ('hidden_units_g', 'seq_length', 'batch_size',
                                              'num_generated_features', 'cond_dim', 'learn_scale')}
        ds = {key: settings[key] for key in ('hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean')}
        d_loss, g_loss = author.GAN_loss(Z, X, gs, ds, settings['kappa'], settings['cond_dim'] > 0,
                                       CG, CD, CS, wrong_labels=settings['wrong_labels'])
        d_solver, g_solver, accountant = author.GAN_solvers(d_loss, g_loss, settings['learning_rate'],
                        settings['batch_size'], samples['train'].shape[0], settings['l2norm_bound'],
                        settings['batches_per_lot'], settings['dp_sigma'], dp=settings['dp'])
        generated = author.generator(Z, **gs, reuse=True)
        fetches = [generated, d_loss, g_loss]
        feeds = {X: samples['train'][:settings['batch_size']],
                 Z: author.sample_Z(settings['batch_size'], settings['seq_length'], settings['latent_dim'], settings['use_time'])}
        data_record = dict(full_split_shapes=expected, settings=settings,
                           data_kind='full_original_source_GP_distribution_not_a_smoke_subset',
                           historical_paper_recipe_certified=False)
    else:
        install_midi_alias()
        import rnn_gan as author
        # Parse flags once; all choices below are registered before execution.
        author.FLAGS(['audit_adopted_gan_native_v1'])
        for key, value in case['flags'].items():
            setattr(author.FLAGS, key, value)
        with tf.variable_scope('model'):
            network = author.RNNGAN(is_training=True, num_song_features=case['num_song_features'],
                                   num_meta_features=case['num_meta_features'])
        fetches = [tf.transpose(tf.stack(network._generated_features), perm=[1, 0, 2])]
        feeds = {}
        data_record = dict(flags=case['flags'], num_song_features=case['num_song_features'],
                           num_meta_features=case['num_meta_features'], input_data_supplied=False,
                           paper_hidden_width=350, historical_paper_recipe_certified=False)
    variables = [dict(name=v.name, shape=v.shape.as_list(), parameters=int(np.prod(v.shape.as_list())))
                 for v in tf.trainable_variables()]
    config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2,
                           inter_op_parallelism_threads=1)
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        actual = session.run(fetches, feed_dict=feeds)
    if not all(np.isfinite(value).all() for value in actual):
        raise ValueError('Full original model initialization/forward must be finite')
    if list(actual[0].shape) != case['expected_generated_shape']:
        raise ValueError('Complete original model generated shape differs')
    return dict(full_native_graph_constructed=True, full_native_initialization_and_forward=True,
                model_parameters=sum(v['parameters'] for v in variables), trainable_variables=variables,
                graph_operations=len(tf.get_default_graph().get_operations()), data=data_record,
                generated_shape=list(actual[0].shape), optimizer_updates=0,
                benchmark_performance=False, author_equivalence_certified=False,
                tensorflow=tf.__version__, numpy=np.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if hashlib.sha256(Path(config['source_registry']).read_bytes()).hexdigest() != config['source_registry_sha256']:
        raise ValueError('Registered original source inventory changed')
    if Path(sys.executable).resolve() != Path(config['python']).resolve():
        raise ValueError('Use the exact registered isolated legacy interpreter')
    case = next(case for case in config['cases'] if case['id'] == args.case)
    acquisition = read(config['source_registry'])
    asset = next(asset for asset in acquisition['assets'] if asset['id'] == case['method'])
    report = Path(config['output_directory']) / (case['id'] + '.json')
    if report.exists():
        raise FileExistsError('Preserve every native preparation attempt')
    before = verify_source(asset)
    status = dict(case=case, actual_python=sys.executable, python_version=sys.version,
                  source_commit=asset['commit'], source_registry=config['source_registry'],
                  training_started=False, benchmark_performance=False, all_original_experiments_complete=False)
    try:
        status['evidence'] = (midi_audit(asset, case, report.parent) if case['kind'] == 'all_midi'
                              else graph_audit(asset, case, report.parent))
        status['passed'] = True
    except Exception as exc:
        status.update(passed=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    after = verify_source(asset)
    status.update(all_pinned_original_files_verified=len(before), source_and_raw_files_unchanged=before == after)
    if before != after:
        raise ValueError('Original source/raw bytes changed')
    write_new(report, status)
    print(json.dumps(dict(case=case['id'], passed=status['passed'], source_unchanged=before == after)), flush=True)
    if not status['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
