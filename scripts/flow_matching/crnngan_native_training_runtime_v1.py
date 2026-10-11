"""Run the complete reviewed author source on every released MIDI composition."""
import os
import hashlib
from pathlib import Path
import random
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
from scripts.flow_matching.crnngan_reviewed_source_v1 import instrumented_main


def run(queue, job):
    import numpy as np
    import tensorflow as tf
    from scripts.flow_matching.audit_adopted_gan_native_v2 import install_legacy_velocity_property
    from scripts.flow_matching.run_timegan_native_job_v1 import flatten_fetches
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen complete native training inputs changed')
    if Path(sys.executable).resolve() != Path(queue['python']).resolve() or tf.__version__ != '1.15.0' or np.__version__ != '1.18.5':
        raise ValueError('Designated native interpreter/numerical packages required')
    import pkg_resources
    package_root = (Path(sys.prefix) / 'Lib/site-packages').resolve()
    packages = {p.key: p.version for p in pkg_resources.working_set if Path(p.location).resolve() == package_root}
    if packages != queue['frozen_native_packages'] or sys.version != queue['frozen_python_version']:
        raise ValueError('Complete frozen original numerical environment required')
    for item in queue['native_interpreter_receipts']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen native interpreter binary changed')
    registry = read(ROOT / queue['source_registry'])
    asset = next(a for a in registry['assets'] if a['id'] == 'crnngan')
    original_receipt = verify_source(asset)
    output = ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every original full training attempt')
    output.mkdir(parents=True)
    source = Path(asset['original_directory'])
    data = source / queue['source_data_subdirectory']
    if not (data / 'do-not-redownload.txt').is_file():
        raise ValueError('Original release must already suppress raw-data downloads')
    install_legacy_velocity_property()
    sys.path.insert(0, str(source))
    import rnn_gan as native
    import music_data_utils
    native.FLAGS.mark_as_parsed()
    for key, value in queue['flags'].items():
        setattr(native.FLAGS, key, value)
    native.FLAGS.tones_per_cell = job['tones_per_cell']
    for key, value in job['variant_flags'].items():
        setattr(native.FLAGS, key, value)
    native.FLAGS.datadir = str(data)
    native.FLAGS.traindir = output.as_posix()
    np.random.seed(job['seed'])
    random.seed(job['seed'])
    # Original held-out filenames use POSIX separators; Windows join uses '\\'.
    original_lists = {key: list(value) for key, value in music_data_utils.file_list.items()}
    music_data_utils.file_list = {key: [os.path.normpath(p) for p in value] for key, value in original_lists.items()}
    loader_class, rnn_class, session_class = music_data_utils.MusicDataLoader, native.RNNGAN, tf.Session
    observed, models, epochs = {}, [], []
    counters = dict(pretraining=0, G=0, D=0)
    operation_ids = {}

    class FullLoader(loader_class):
        def __init__(self, *args, **kwargs):
            self.native_files = {}
            loader_class.__init__(self, *args, **kwargs)
            split = {}
            for key, songs in self.songs.items():
                split[key] = [self.native_files[id(song[2])] for song in songs]
            all_files = [v for values in split.values() for v in values]
            if len(all_files) != queue['parsed_released_files'] or sum(v['note_events'] for v in all_files) != queue['full_note_events']:
                raise ValueError('Every originally parsed released composition/note event required')
            names = [v['path'] for v in all_files]
            if len(names) != len(set(names)) or any(len(values) < native.FLAGS.batch_size for values in split.values()):
                raise ValueError('Disjoint complete nonempty source filename holdouts required')
            self.full_split = split
            if verify_source(asset) != original_receipt:
                raise ValueError('Complete original raw inputs changed during full loader traversal')
            observed['loader'] = self
            write(output / 'full_original_loader_audit.json', dict(full_parsed_compositions=len(all_files),
                full_note_events=sum(v['note_events'] for v in all_files), filename_splits=split,
                split_counts={key: len(values) for key, values in split.items()},
                original_heldout_membership_preserved_after_windows_separator_normalization=True,
                original_paper_historical_corpus_identity_certified=False, raw_files_unchanged=True,
                full_source_receipt=original_receipt))

        def read_one_file(self, directory, filename, pace_events):
            result = loader_class.read_one_file(self, directory, filename, pace_events)
            if result is not None:
                self.native_files[id(result)] = dict(path=(Path(directory) / filename).relative_to(data).as_posix(), note_events=len(result))
            return result

    class ObservedSession(session_class):
        def __init__(self, *args, **kwargs):
            kwargs['config'] = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
            session_class.__init__(self, *args, **kwargs)
            observed['session'] = self

        def run(self, fetches, *args, **kwargs):
            result = session_class.run(self, fetches, *args, **kwargs)
            for fetched in flatten_fetches(fetches):
                key = operation_ids.get(id(fetched))
                if key:
                    counters[key] += 1
            return result

    class CompleteRNN(rnn_class):
        def __init__(self, *args, **kwargs):
            previous = set(v.name for v in tf.global_variables())
            previous_trainable = set(v.name for v in tf.trainable_variables())
            rnn_class.__init__(self, *args, **kwargs)
            if previous:
                fresh = [v for v in tf.global_variables() if v.name not in previous]
                if set(v.name for v in tf.trainable_variables()) != previous_trainable or len(fresh) != 1 or fresh[0] is not self._lr:
                    raise ValueError('Only the original newly rebuilt scalar learning rate may need initialization')
                # Source tf.Variable creates a fresh uninitialized learning rate at curriculum rebuild.
                observed['session'].run(tf.variables_initializer(fresh))
            for key, attribute in [('pretraining', 'opt_pretraining'), ('G', 'opt_g'), ('D', 'opt_d')]:
                if hasattr(self, attribute):
                    operation_ids[id(getattr(self, attribute))] = key
            models.append(dict(songlength=self.songlength, trainable_parameters=sum(int(np.prod(v.shape.as_list())) for v in tf.trainable_variables()),
                               fresh_scalar_learning_rate_initialized=bool(previous)))

    def graph_seed():
        tf.set_random_seed(job['seed'])

    def epoch_complete(epoch, local):
        if epoch != len(epochs):
            raise ValueError('Every full original epoch must be observed in sequence')
        session, loader = observed['session'], observed['loader']
        if loader.pointer['train'] // native.FLAGS.batch_size != len(loader.songs['train']) // native.FLAGS.batch_size:
            if not (local['train_g_loss'] == 0 and local['train_d_loss'] == 0):
                raise ValueError('Do not shorten the original full training composition traversal')
        values = [local[k] for k in ['train_g_loss', 'train_d_loss', 'valid_g_loss', 'valid_d_loss'] if local[k] is not None]
        if not np.isfinite(values).all():
            raise ValueError('Full original train/validation losses became nonfinite')
        directory = output / 'native_checkpoints' / ('epoch_%03d' % epoch)
        directory.mkdir(parents=True)
        variables = tf.global_variables()
        if observed.get('saver_membership') != tuple(v.name for v in variables):
            observed['native_saver'] = tf.train.Saver(var_list=variables, max_to_keep=None)
            observed['saver_membership'] = tuple(v.name for v in variables)
            write_meta = True
        else:
            write_meta = False
        observed['native_saver'].save(session, str(directory / 'model'), write_meta_graph=write_meta)
        if write_meta:
            observed['latest_meta_graph'] = str(directory / 'model.meta')
        reader = tf.train.NewCheckpointReader(str(directory / 'model'))
        live = session.run(variables)
        tensors = []
        for variable, array in zip(variables, live):
            name = variable.name.rsplit(':', 1)[0]
            restored = reader.get_tensor(name)
            if not np.isfinite(array).all() or not np.array_equal(array, restored):
                raise ValueError('Every full native variable must be finite and exactly match its checkpoint')
            tensors.append(dict(name=name, shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()))
        write(directory / 'full_checkpoint_variable_audit.json', dict(tensors=tensors,
            full_variable_count=len(tensors), all_live_variables_bitwise_equal_to_checkpoint=True,
            meta_graph=observed['latest_meta_graph'], meta_graph_sha256=sha(observed['latest_meta_graph']),
            stochastic_TensorFlow_RNG_exact_resume_certified=False))
        state = np.random.get_state()
        np.savez_compressed(str(directory / 'numpy_random_state.npz'), algorithm=state[0], keys=state[1], position=state[2], has_gauss=state[3], cached_gaussian=state[4])
        write(directory / 'python_random_state.json', random.getstate())
        epochs.append(dict(epoch=epoch, songlength=native.FLAGS.songlength, optimizer_updates=dict(counters),
            losses={k: None if local[k] is None else float(local[k]) for k in ['train_g_loss', 'train_d_loss', 'valid_g_loss', 'valid_d_loss']},
            full_training_batches=len(loader.songs['train']) // native.FLAGS.batch_size,
            checkpoint_artifacts=[dict(path=str(p), sha256=sha(p)) for p in directory.iterdir() if p.is_file()]))
        write(output / 'progress.json', dict(job=job, epochs_completed=len(epochs), required_epochs=500,
            epochs=epochs, model_constructions=models, optimizer_updates=counters,
            full_training_completed=False, stochastic_TensorFlow_RNG_exact_resume_certified=False,
            author_equivalence_certified=False, benchmark_performance=False))

    def test_complete(local):
        loader = observed['loader']
        if loader.pointer['test'] // native.FLAGS.batch_size != len(loader.songs['test']) // native.FLAGS.batch_size:
            raise ValueError('Every original full test minibatch required')
        losses = dict(G=float(local['test_g_loss']), D=float(local['test_d_loss']))
        if not np.isfinite(list(losses.values())).all():
            raise ValueError('Complete original final test losses must be finite')
        observed['test'] = dict(losses=losses, full_batches=len(loader.songs['test']) // native.FLAGS.batch_size,
            full_compositions=len(loader.songs['test']), source_discards_incomplete_tail=True)
        write(output / 'full_original_test_audit.json', observed['test'])

    music_data_utils.MusicDataLoader = FullLoader
    native.RNNGAN = CompleteRNN
    tf.Session = ObservedSession
    namespace = dict(vars(native), _native_graph_seed=graph_seed, _native_epoch_complete=epoch_complete,
                     _native_test_complete=test_complete)
    exec(compile(instrumented_main(source / 'rnn_gan.py'), str(source / 'rnn_gan.py'), 'exec'), namespace)
    namespace['main'](None)
    if len(epochs) != 500 or verify_source(asset) != original_receipt:
        raise ValueError('All500 full epochs and unchanged original source/raw inputs required')
    write(output / 'result.json', dict(job=job, epochs=epochs, full500_epochs_completed=True,
        full_training_completed=True, actual_optimizer_updates=counters, model_constructions=models,
        full_data_audit=read(output / 'full_original_loader_audit.json'), reviewed_source_corrections=queue['reviewed_corrections'],
        full_test_audit=observed['test'],
        original_source_and_raw_data_unchanged=True, native_stochastic_checkpoint_replay_pending=True,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
