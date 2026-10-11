"""Verify four complete native epochs, including unchanged graph RNG seeds."""
import argparse
import ast
import os
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import NATIVE_ENVIRONMENT


class FourCompleteEpochsStop(Exception):
    pass


def worker(config):
    import tensorflow as tf
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    from scripts.flow_matching.crnngan_executor_cache_runtime_v1 import run
    from scripts.flow_matching.audit_crnngan_checkpoint_variables_v1 import verify_tensors
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen complete native four-epoch input changed')
    queue = read(ROOT/config['queue'])
    reference = read(config['reference_progress'])
    if config['required_complete_epochs'] != 4 or len(reference['epochs']) != 4:
        raise ValueError('All four preserved complete native epochs required')
    job = dict(queue['jobs'][0], output_directory=config['native_attempt_directory'])
    original_write = original.write
    reports = []

    def graph_metadata(path):
        graph = tf.MetaGraphDef()
        graph.ParseFromString(Path(path).read_bytes())
        return dict(operations=len(graph.graph_def.node), no_ops=sum(n.op == 'NoOp' for n in graph.graph_def.node),
            seeded_operations=[dict(name=n.name, operation=n.op, seed=n.attr['seed'].i, seed2=n.attr['seed2'].i)
                               for n in graph.graph_def.node if 'seed' in n.attr and 'seed2' in n.attr])

    def observe_write(path, progress):
        original_write(path, progress)
        if Path(path).name != 'progress.json':
            return
        index = progress['epochs_completed']-1
        if index != len(reports) or index >= 4:
            raise ValueError('Every complete native epoch must be audited in sequence')
        expected = reference['epochs'][index]
        actual = progress['epochs'][index]
        directory = Path(job['output_directory'])/'native_checkpoints'/('epoch_%03d'%index)
        previous = Path(reference['job']['output_directory'])/'native_checkpoints'/('epoch_%03d'%index)
        old_audit, new_audit = read(previous/'full_checkpoint_variable_audit.json'), read(directory/'full_checkpoint_variable_audit.json')
        if old_audit['tensors'] != new_audit['tensors'] or actual['optimizer_updates'] != expected['optimizer_updates']:
            raise ValueError('Full native epoch state or optimizer budget differs at epoch '+str(index))
        parameters = verify_tensors(tf.train.NewCheckpointReader(str(directory/'model')), new_audit)
        old_graph, new_graph = graph_metadata(old_audit['meta_graph']), graph_metadata(new_audit['meta_graph'])
        if old_graph != new_graph:
            raise ValueError('Original whole graph geometry or actual random operation seeds differ')
        reports.append(dict(epoch=index, songlength=actual['songlength'], full_training_batches=actual['full_training_batches'],
            optimizer_updates=actual['optimizer_updates'], full_native_variables=len(new_audit['tensors']),
            all_native_variable_hashes_bitwise_identical=True, whole_graph_geometry_and_rng_seeds_identical=True,
            trainable_parameters=parameters, original_losses=expected['losses'], actual_losses=actual['losses'],
            loss_scalar_bitwise_identical=expected['losses'] == actual['losses'], original_graph=old_graph,
            native_meta_graph_sha256=sha(new_audit['meta_graph'])))
        write(ROOT/config['output_directory']/'verified_complete_epochs.json', dict(records=reports,
            required_complete_epochs=4, full500_training_completed=False, benchmark_performance=False))
        if len(reports) == 4:
            raise FourCompleteEpochsStop()

    original.write = observe_write
    try:
        run(queue, job)
    except FourCompleteEpochsStop:
        pass
    finally:
        original.write = original_write
    if len(reports) != 4:
        raise ValueError('All four complete epochs/checkpoints must pass')
    write(ROOT/config['output_directory']/'four_complete_epoch_executor_audit.json', dict(
        full_complete_epochs_verified=4, all_full_native_train_states_bitwise_identical=True,
        all_original_graph_geometry_and_random_seeds_identical=True, records=reports,
        scalar_loss_bitwise_equivalence_certified=False, full500_training_completed=False,
        final_full100_curriculum_capacity_verified=False, benchmark_performance=False,
        author_equivalence_certified=False, all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    if config['native_execution_environment'] != NATIVE_ENVIRONMENT:
        raise ValueError('Original CPU/BLAS execution required')
    if args.worker:
        if any(os.environ.get(k) != v for k, v in NATIVE_ENVIRONMENT.items()):
            raise ValueError('Frozen original native environment differs')
        worker(config)
        return
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import host_execution_tree
    from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base
    tree = host_execution_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v4',
                                            'scripts.flow_matching.audit_crnngan_four_complete_epochs_v1')
            if node.value == 'full_epoch_noop_equivalence.json':
                node.value = 'four_complete_epoch_executor_audit.json'
    exec(compile(tree, base.__file__, 'exec'), dict(__name__='__main__', __file__=base.__file__, os=os))


if __name__ == '__main__':
    main()
