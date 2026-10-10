"""Execute untouched original TimeGAN main, observing its complete native budget.

Only filesystem output, CPU threads, graph seeds and observation are supplied
externally. No network/loss/optimizer/data loop or evaluator source is replaced.
"""
import argparse
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha, verify_queue, verify_counts


def flatten_fetches(value):
    if isinstance(value, dict):
        return [v for x in value.values() for v in flatten_fetches(x)]
    if isinstance(value, (list, tuple)):
        return [v for x in value for v in flatten_fetches(x)]
    return [value]


def validate_evaluators(records):
    for kind, updates in [('discriminative', 2000), ('predictive', 5000)]:
        selected = [r for r in records if r['kind'] == kind]
        if len(selected) != 10 or any(r['optimizer_updates'] != updates for r in selected):
            raise ValueError('All ten full original posthoc evaluator trainings required')
        if [r['repeat'] for r in selected] != list(range(10)):
            raise ValueError('Every frozen native evaluator repeat is required')
        if any(not __import__('math').isfinite(r['score']) for r in selected):
            raise ValueError('Every complete original evaluator score must be finite')


def run(queue, job, queue_path):
    model = verify_queue(queue)
    environment = read(ROOT / queue['environment_lock'])
    runtime = Path(environment['runtime_root'])
    for name, key in [('python.exe', 'python_executable_sha256'), ('python37.dll', 'python37_dll_sha256'),
                      ('python37._pth', 'python_pth_sha256')]:
        if sha(runtime / name) != environment[key]:
            raise ValueError('Frozen native interpreter changed')
    output = ROOT / job['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every original full-budget job attempt')
    output.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    source = ROOT / model['source_root']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import tensorflow as tf
    import main_timegan as author
    import matplotlib.pyplot as plt
    import pkg_resources
    package_root = (Path(environment['runtime_root']) / 'Lib/site-packages').resolve()
    actual_packages = {p.key: p.version for p in pkg_resources.working_set
                       if Path(p.location).resolve() == package_root}
    if (sys.version != environment['python'] or tf.__version__ != '1.15.0'
            or actual_packages != environment['packages']):
        raise ValueError('Frozen exact native TF1 numerical environment differs')
    np.random.seed(job['seed'])
    expected_shape = read(ROOT / model['dataset_config'])['datasets'][job['dataset']]['full_shape']
    graph_audit = read(ROOT / queue['construction_proof'])
    audited_model = next(r for r in graph_audit['records']
                         if r['dataset'] == job['dataset'] and r['module'] == job['module'])
    context = dict(kind=None, repeat=None, session=None, graph_seeds=[], completed_evaluators=[])
    original_reset = tf.reset_default_graph
    original_session = tf.Session
    checkpoints = []
    model_counts = None
    model_parameters = None

    def reset_graph():
        original_reset()
        index = len(context['graph_seeds'])
        seed = job['seed'] + 1009 * index
        tf.set_random_seed(seed)
        context['graph_seeds'].append(dict(index=index, kind=context['kind'], repeat=context['repeat'], seed=seed))

    def progress(session):
        write(output / 'progress.json', dict(job=job, captured_utc=datetime.now(timezone.utc).isoformat(),
              phase=session.kind, evaluator_repeat=context['repeat'], optimizer_updates=session.counts,
              full_original_training_complete=False, full_evaluation_complete=False))

    class ObservedSession(original_session):
        def __init__(self, *args, **kwargs):
            local = dict(inspect.currentframe().f_back.f_locals)
            self.kind = context['kind']
            self.native_nodes = local
            self.ops = ({name: local[name + '_solver'] for name in ('E0', 'E', 'D', 'G', 'GS')}
                        if self.kind == 'timegan' else {'evaluator': local['d_solver' if self.kind == 'discriminative' else 'p_solver']})
            self.counts = {name: 0 for name in self.ops}
            self.recording = False
            self.saver = None
            self.parameters = int(sum(np.prod(v.shape.as_list()) for v in tf.trainable_variables()))
            if self.kind == 'timegan' and self.parameters != audited_model['model_parameters']:
                raise ValueError('Actual full original network differs from the registered full graph')
            kwargs['config'] = tf.ConfigProto(device_count={'GPU': 0},
                  intra_op_parallelism_threads=queue['cpu_threads'], inter_op_parallelism_threads=1)
            super(ObservedSession, self).__init__(*args, **kwargs)
            context['session'] = self

        def run(self, fetches, *args, **kwargs):
            result = super(ObservedSession, self).run(fetches, *args, **kwargs)
            if self.recording:
                return result
            fetched = flatten_fetches(fetches)
            changed = False
            for name, op in self.ops.items():
                if any(value is op for value in fetched):
                    self.counts[name] += 1
                    changed = True
            if changed and sum(self.counts.values()) % 1000 == 0:
                progress(self)
            if self.kind == 'timegan':
                if changed and self.counts['E0'] == 50000 and self.counts['GS'] == 0:
                    self.save_native('embedding_complete')
                if changed and self.counts['GS'] == 50000 and self.counts['G'] == 0:
                    self.save_native('supervised_complete')
                if fetches is self.native_nodes['X_hat']:
                    feed = kwargs['feed_dict']
                    if list(np.asarray(result).shape) != expected_shape or not np.isfinite(result).all():
                        raise ValueError('Generation must cover every original finite sample/window')
                    np.savez_compressed(output / 'full_native_generation_feed.npz',
                        noise=np.asarray(feed[self.native_nodes['Z']]),
                        normalized_real=np.asarray(feed[self.native_nodes['X']]),
                        times=np.asarray(feed[self.native_nodes['T']]),
                        normalized_generated=np.asarray(result),
                        min_val=self.native_nodes['min_val'], max_val=self.native_nodes['max_val'])
            elif self.kind == 'predictive' and fetches is self.native_nodes['y_pred']:
                feed = kwargs['feed_dict']
                if np.asarray(result).shape[0] != expected_shape[0]:
                    raise ValueError('Native predictor must evaluate all original windows')
                np.savez_compressed(output / ('predictive_%02d_full_test.npz' % context['repeat']),
                    predictions=result, targets=np.asarray(feed[self.native_nodes['Y']]))
            elif self.kind == 'discriminative' and isinstance(fetches, list) and len(fetches) == 2:
                if (fetches[0] is self.native_nodes['y_pred_real'] and fetches[1] is self.native_nodes['y_pred_fake']):
                    test_count = expected_shape[0] - int(expected_shape[0] * .8)
                    if any(len(v) != test_count for v in result):
                        raise ValueError('Both complete original native 20% discriminator test splits required')
                    np.savez_compressed(output / ('discriminative_%02d_full_test.npz' % context['repeat']),
                                        real_probabilities=result[0], synthetic_probabilities=result[1])
            return result

        def save_native(self, tag):
            self.recording = True
            try:
                if self.saver is None:
                    self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
                directory = output / 'checkpoints' / tag
                directory.mkdir(parents=True)
                prefix = directory / 'model'
                self.saver.save(self, str(prefix))
                state = np.random.get_state()
                np.savez_compressed(directory / 'numpy_random_state.npz', algorithm=state[0], keys=state[1],
                                    position=state[2], has_gauss=state[3], cached_gaussian=state[4])
                records = [dict(path=str(p), sha256=sha(p)) for p in directory.iterdir() if p.is_file()]
                checkpoints.append(dict(tag=tag, kind=self.kind, optimizer_updates=dict(self.counts),
                                        model_parameters=self.parameters, artifacts=records))
                write(directory / 'receipt.json', checkpoints[-1])
            finally:
                self.recording = False

    tf.reset_default_graph = reset_graph
    tf.Session = ObservedSession
    native_timegan = author.timegan

    def complete_timegan(data, parameters):
        nonlocal model_counts, model_parameters
        context.update(kind='timegan', repeat=None, session=None)
        if list(np.asarray(data).shape) != expected_shape:
            raise ValueError('Original complete native data shape required')
        generated = native_timegan(data, parameters)
        session = context['session']
        verify_counts(session.counts, model)
        session.save_native('timegan_complete')
        model_counts, model_parameters = dict(session.counts), session.parameters
        session.close()
        np.savez_compressed(output / 'full_native_real_and_generated.npz', real=np.asarray(data), generated=np.asarray(generated))
        return generated

    author.timegan = complete_timegan
    for kind in ('discriminative', 'predictive'):
        function_name = kind + '_score_metrics'
        original = getattr(author, function_name)

        def evaluator(data, generated, original=original, kind=kind):
            repeat = sum(r['kind'] == kind for r in context['completed_evaluators'])
            context.update(kind=kind, repeat=repeat, session=None)
            score = float(original(data, generated))
            session = context['session']
            expected_updates = 2000 if kind == 'discriminative' else 5000
            if session.counts != {'evaluator': expected_updates}:
                raise ValueError('Full native evaluator budget required')
            session.save_native('%s_%02d_complete' % (kind, repeat))
            session.close()
            context['completed_evaluators'].append(dict(kind=kind, repeat=repeat, score=score,
                  optimizer_updates=expected_updates, graph_seed=context['graph_seeds'][-1]['seed']))
            write(output / 'completed_evaluators.json', context['completed_evaluators'])
            return score

        setattr(author, function_name, evaluator)
    figures = []

    def save_figure():
        path = output / ('native_visualization_%d.png' % len(figures))
        plt.savefig(str(path), dpi=150)
        plt.close()
        figures.append(dict(path=str(path), sha256=sha(path)))

    plt.show = save_figure
    parameters = model['parameters']
    args = SimpleNamespace(data_name=job['dataset'], module=job['module'], seq_len=parameters['seq_len'],
        hidden_dim=parameters['hidden_dim'], num_layer=parameters['num_layer'], iteration=parameters['iterations'],
        batch_size=parameters['batch_size'], metric_iteration=parameters['metric_iteration'])
    real, generated, scores = author.main(args)
    validate_evaluators(context['completed_evaluators'])
    if len(figures) != 2 or len(checkpoints) != 23 or len(context['graph_seeds']) != 21:
        raise ValueError('Original two visualizations, full model/stage and all twenty evaluator checkpoints required')
    artifacts = [dict(path=str(p), sha256=sha(p)) for p in output.glob('*.npz')]
    if len(artifacts) != 22:
        raise ValueError('Full generation/feed and every native evaluator prediction array required')
    write(output / 'result.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), job=job,
        queue_sha256=sha(ROOT / queue_path), environment_lock_sha256=sha(ROOT / queue['environment_lock']),
        full_original_source_main_completed=True, full_native_three_training_stages_completed=True,
        full_native_evaluator_trainings_completed=True, optimizer_updates=model_counts,
        full_original_data_shape=list(np.asarray(real).shape), full_generated_shape=list(np.asarray(generated).shape),
        model_parameters=model_parameters, scores={k: float(v) for k, v in scores.items()},
        evaluators=context['completed_evaluators'], graph_seeds=context['graph_seeds'], checkpoints=checkpoints,
        artifacts=artifacts, visualizations=figures, independent_model_training_seed_slots=1,
        independent_full_checkpoint_replay_completed=False, author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False,
        boundary=queue['boundary']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(queue, job, args.queue)


if __name__ == '__main__':
    main()
