"""Independently compare every original native graph node and function definition."""
import argparse
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def compare_graph_definitions(left, right):
    old = {n.name: n.SerializeToString(deterministic=True) for n in left.node}
    new = {n.name: n.SerializeToString(deterministic=True) for n in right.node}
    if old != new:
        names = sorted(set(old) | set(new))
        differing = [name for name in names if old.get(name) != new.get(name)]
        raise ValueError('Original native graph node definitions differ: '+str(differing[:20]))
    if (left.library.SerializeToString(deterministic=True) != right.library.SerializeToString(deterministic=True)
            or left.versions.SerializeToString(deterministic=True) != right.versions.SerializeToString(deterministic=True)):
        raise ValueError('Original native function library or graph version differs')
    return len(old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT/args.config)
    import tensorflow as tf
    proof = read(ROOT/config['output_directory']/'four_complete_epoch_executor_audit.json')
    if proof['full_complete_epochs_verified'] != 4:
        raise ValueError('Every complete native epoch required')
    previous = Path(read(config['reference_progress'])['job']['output_directory'])
    actual = Path(config['native_attempt_directory'])
    records = []
    for epoch in range(4):
        old = read(previous/'native_checkpoints'/('epoch_%03d'%epoch)/'full_checkpoint_variable_audit.json')
        new = read(actual/'native_checkpoints'/('epoch_%03d'%epoch)/'full_checkpoint_variable_audit.json')
        graphs = []
        for audit in [old, new]:
            if sha(audit['meta_graph']) != audit['meta_graph_sha256']:
                raise ValueError('Complete source graph file changed')
            graph = tf.MetaGraphDef()
            graph.ParseFromString(Path(audit['meta_graph']).read_bytes())
            graphs.append(graph.graph_def)
        count = compare_graph_definitions(*graphs)
        records.append(dict(epoch=epoch, full_original_graph_node_count=count,
            every_native_node_definition_bitwise_identical=True, complete_function_library_identical=True,
            original_meta_graph_sha256=old['meta_graph_sha256'], native_meta_graph_sha256=new['meta_graph_sha256']))
    write(ROOT/config['output_directory']/'independent_all_graph_nodes.json', dict(
        full_original_epochs=4, all_complete_native_graph_definitions_bitwise_identical=True,
        records=records, original_random_seed_operation_ids_preserved=True,
        full500_training_completed=False, benchmark_performance=False))


if __name__ == '__main__':
    main()
