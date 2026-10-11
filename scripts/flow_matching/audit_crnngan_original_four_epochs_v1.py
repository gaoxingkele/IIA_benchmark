"""Measure native four-epoch repeatability with the original Session executor."""
import argparse
import ast
import os
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import NATIVE_ENVIRONMENT


class FourOriginalEpochsStop(Exception):
    pass


def compare_arrays(previous, actual, names, np):
    differences = []
    for name in names:
        old, new = previous(name), actual(name)
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError('Native checkpoint variable geometry differs: ' + name)
        if not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('Native checkpoint contains nonfinite values: ' + name)
        if not np.array_equal(old, new):
            differences.append(dict(name=name, max_absolute_difference=float(
                np.max(np.abs(old.astype(np.float64) - new.astype(np.float64))))))
    return differences


def worker(config):
    import numpy as np
    import tensorflow as tf
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    from scripts.flow_matching.audit_crnngan_checkpoint_variables_v1 import verify_tensors
    from scripts.flow_matching.audit_crnngan_complete_graph_nodes_v1 import compare_graph_definitions
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen original-executor comparison input changed')
    queue = read(ROOT / config['queue'])
    reference = read(config['reference_progress'])
    if config['required_complete_epochs'] != 4 or len(reference['epochs']) != 4:
        raise ValueError('All four preserved complete native epochs required')
    job = dict(queue['jobs'][0], output_directory=config['native_attempt_directory'])
    original_write = original.write
    reports = []

    def observe_write(path, progress):
        original_write(path, progress)
        if Path(path).name != 'progress.json':
            return
        index = progress['epochs_completed'] - 1
        if index != len(reports) or index >= 4:
            raise ValueError('Every full original epoch must be observed in sequence')
        expected, actual = reference['epochs'][index], progress['epochs'][index]
        if (actual['optimizer_updates'] != expected['optimizer_updates']
                or actual['full_training_batches'] != expected['full_training_batches']
                or actual['songlength'] != expected['songlength']):
            raise ValueError('Original full data traversal, curriculum or update budget differs')
        relative = Path('native_checkpoints') / ('epoch_%03d' % index)
        previous = Path(reference['job']['output_directory']) / relative
        directory = Path(job['output_directory']) / relative
        old_audit = read(previous / 'full_checkpoint_variable_audit.json')
        new_audit = read(directory / 'full_checkpoint_variable_audit.json')
        old_reader = tf.train.NewCheckpointReader(str(previous / 'model'))
        new_reader = tf.train.NewCheckpointReader(str(directory / 'model'))
        parameters = verify_tensors(old_reader, old_audit)
        if verify_tensors(new_reader, new_audit) != parameters or parameters != 4924855:
            raise ValueError('Original complete network must remain unchanged')
        names = [item['name'] for item in old_audit['tensors']]
        if names != [item['name'] for item in new_audit['tensors']]:
            raise ValueError('Complete original variable membership differs')
        differences = compare_arrays(old_reader.get_tensor, new_reader.get_tensor, names, np)
        graphs = []
        for audit in (old_audit, new_audit):
            if sha(audit['meta_graph']) != audit['meta_graph_sha256']:
                raise ValueError('Native graph file changed')
            graph = tf.MetaGraphDef()
            graph.ParseFromString(Path(audit['meta_graph']).read_bytes())
            graphs.append(graph.graph_def)
        nodes = compare_graph_definitions(*graphs)
        with np.load(str(previous / 'numpy_random_state.npz'), allow_pickle=False) as old_rng:
            with np.load(str(directory / 'numpy_random_state.npz'), allow_pickle=False) as new_rng:
                numpy_equal = set(old_rng.files) == set(new_rng.files) and all(
                    np.array_equal(old_rng[key], new_rng[key]) for key in old_rng.files)
        midi_paths = []
        for root in (previous.parent.parent, directory.parent.parent):
            paths = list((root / 'generated_data').glob('out-seed%d-%d-*.mid' % (job['seed'], index)))
            if len(paths) != 1:
                raise ValueError('One complete original generated MIDI sample required')
            midi_paths.append(paths[0])
        reports.append(dict(epoch=index, songlength=actual['songlength'],
            full_training_batches=actual['full_training_batches'], optimizer_updates=actual['optimizer_updates'],
            full_native_variables=len(names), full_trainable_parameters=parameters,
            variable_differences=differences, all_variables_bitwise_identical=not differences,
            full_native_graph_nodes=nodes, all_graph_nodes_and_library_bitwise_identical=True,
            numpy_rng_state_identical=numpy_equal,
            python_rng_state_identical=read(previous / 'python_random_state.json') == read(directory / 'python_random_state.json'),
            generated_midi_bitwise_identical=sha(midi_paths[0]) == sha(midi_paths[1]),
            original_losses=expected['losses'], actual_losses=actual['losses'],
            original_meta_graph_sha256=old_audit['meta_graph_sha256'],
            actual_meta_graph_sha256=new_audit['meta_graph_sha256']))
        write(ROOT / config['output_directory'] / 'original_epoch_comparisons.json', dict(
            records=reports, required_complete_epochs=4, full500_training_completed=False,
            benchmark_performance=False))
        if len(reports) == 4:
            raise FourOriginalEpochsStop()

    original.write = observe_write
    try:
        # Use the original frozen runtime directly; do not install either caching candidate.
        original.run(queue, job)
    except FourOriginalEpochsStop:
        pass
    finally:
        original.write = original_write
    if len(reports) != 4:
        raise ValueError('All four full native epochs must actually execute')
    write(ROOT / config['output_directory'] / 'original_four_epoch_repeatability.json', dict(
        full_complete_epochs_executed=4, original_executor_unchanged=True,
        all_original_data_curriculum_and_optimizer_budgets_identical=True,
        all_complete_graph_definitions_bitwise_identical=True,
        all_trained_states_bitwise_identical=all(r['all_variables_bitwise_identical'] for r in reports),
        records=reports, full500_training_completed=False, benchmark_performance=False,
        author_equivalence_certified=False, all_original_experiments_complete=False))


def host_execution_tree():
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import host_execution_tree as base_tree
    tree = base_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v4',
                                            'scripts.flow_matching.audit_crnngan_original_four_epochs_v1')
            if node.value == 'full_epoch_noop_equivalence.json':
                node.value = 'original_four_epoch_repeatability.json'
            if node.value == 'running_complete_native_epoch_equivalence':
                node.value = 'running_original_four_complete_epochs'
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if config['native_execution_environment'] != NATIVE_ENVIRONMENT:
        raise ValueError('Frozen original CPU/BLAS execution required')
    if args.worker:
        if any(os.environ.get(k) != v for k, v in NATIVE_ENVIRONMENT.items()):
            raise ValueError('Native original CPU/BLAS environment differs')
        worker(config)
        return
    from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base
    exec(compile(host_execution_tree(), base.__file__, 'exec'),
         dict(__name__='__main__', __file__=base.__file__, os=os))


if __name__ == '__main__':
    main()
