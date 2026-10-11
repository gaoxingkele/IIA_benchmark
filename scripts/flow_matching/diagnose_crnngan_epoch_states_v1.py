"""Read complete native checkpoints/graphs/RNG/samples without training changes."""
import argparse
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def diagnose(config):
    import numpy as np
    import tensorflow as tf
    from scripts.flow_matching.audit_crnngan_original_four_epochs_v1 import compare_arrays
    from scripts.flow_matching.audit_crnngan_checkpoint_variables_v1 import verify_tensors
    from scripts.flow_matching.audit_crnngan_complete_graph_nodes_v1 import compare_graph_definitions
    reference = read(config['reference_progress'])
    actual = read(Path(config['native_attempt_directory']) / 'progress.json')
    required = config['required_complete_epochs']
    if required != 4 or len(reference['epochs']) != required or len(actual['epochs']) != required:
        raise ValueError('Both attempts must contain all four complete original epochs')
    records = []
    for epoch in range(required):
        expected, observed = reference['epochs'][epoch], actual['epochs'][epoch]
        if any(expected[k] != observed[k] for k in ('epoch', 'songlength', 'full_training_batches', 'optimizer_updates')):
            raise ValueError('Full original epoch curriculum/data/update budgets differ')
        directories = [Path(progress['job']['output_directory']) / 'native_checkpoints' / ('epoch_%03d' % epoch)
                       for progress in (reference, actual)]
        audits = [read(directory / 'full_checkpoint_variable_audit.json') for directory in directories]
        readers = [tf.train.NewCheckpointReader(str(directory / 'model')) for directory in directories]
        for audit, reader in zip(audits, readers):
            if verify_tensors(reader, audit) != 4924855:
                raise ValueError('Complete original trainable network required')
        names = [item['name'] for item in audits[0]['tensors']]
        if names != [item['name'] for item in audits[1]['tensors']]:
            raise ValueError('Every native variable must be compared')
        differences = compare_arrays(readers[0].get_tensor, readers[1].get_tensor, names, np)
        graphs = []
        for audit in audits:
            if sha(audit['meta_graph']) != audit['meta_graph_sha256']:
                raise ValueError('Native complete graph hash mismatch')
            graph = tf.MetaGraphDef()
            graph.ParseFromString(Path(audit['meta_graph']).read_bytes())
            graphs.append(graph.graph_def)
        nodes = compare_graph_definitions(*graphs)
        with np.load(str(directories[0] / 'numpy_random_state.npz'), allow_pickle=False) as old:
            with np.load(str(directories[1] / 'numpy_random_state.npz'), allow_pickle=False) as new:
                numpy_equal = set(old.files) == set(new.files) and all(np.array_equal(old[k], new[k]) for k in old.files)
        samples = []
        for progress in (reference, actual):
            root = Path(progress['job']['output_directory'])
            found = list((root / 'generated_data').glob('out-%s-%d-*.mid' % (root.name, epoch)))
            if len(found) != 1:
                raise ValueError('One original native generated MIDI sample required')
            samples.append(dict(path=str(found[0]), sha256=sha(found[0])))
        records.append(dict(epoch=epoch, songlength=observed['songlength'],
            full_training_batches=observed['full_training_batches'], optimizer_updates=observed['optimizer_updates'],
            full_variable_count=len(names), complete_native_graph_nodes=nodes,
            all_graph_nodes_and_library_bitwise_identical=True, variable_differences=differences,
            all_variables_bitwise_identical=not differences, numpy_rng_state_identical=numpy_equal,
            python_rng_state_identical=read(directories[0] / 'python_random_state.json') == read(directories[1] / 'python_random_state.json'),
            original_generated_midi_bitwise_identical=samples[0]['sha256'] == samples[1]['sha256'],
            generated_samples=samples, variable_audits=[dict(path=str(d / 'full_checkpoint_variable_audit.json'),
                sha256=sha(d / 'full_checkpoint_variable_audit.json')) for d in directories],
            complete_graph_hashes=[a['meta_graph_sha256'] for a in audits]))
    return dict(records=records, full_original_epochs=required,
        all_complete_native_graph_definitions_bitwise_identical=True,
        all_trained_states_bitwise_identical=all(r['all_variables_bitwise_identical'] for r in records),
        full500_training_completed=False, benchmark_performance=False,
        author_equivalence_certified=False, all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnose(read(ROOT / args.config))
    result['configuration_sha256'] = sha(ROOT / args.config)
    result['diagnostic_script_sha256'] = sha(Path(__file__))
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError('Preserve every prior native diagnostic receipt')
    write(target, result)
    print('Complete native epochs audited: %d; all trained states bitwise identical: %s' % (
        result['full_original_epochs'], result['all_trained_states_bitwise_identical']))


if __name__ == '__main__':
    main()
