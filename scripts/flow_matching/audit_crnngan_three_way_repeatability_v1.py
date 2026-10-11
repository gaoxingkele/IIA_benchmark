"""Compare preserved, repeated-original and cache-executor full native states."""
import argparse
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def require_complete_original_diagnostic(proof, guard):
    if (guard['exit_code'] != 0 or not guard['passed'] or guard['resource_guard_aborted']
            or proof['full_complete_epochs_executed'] != 4
            or not proof['original_executor_unchanged']
            or not proof['all_original_data_curriculum_and_optimizer_budgets_identical']
            or not proof['all_complete_graph_definitions_bitwise_identical']
            or len(proof['records']) != 4
            or [r['epoch'] for r in proof['records']] != list(range(4))):
        raise ValueError('All four complete unchanged original-executor epochs must actually execute')
    for i, record in enumerate(proof['records']):
        if (record['full_training_batches'] != 130
                or record['optimizer_updates'] != dict(pretraining=130*(i+1), G=0, D=0)
                or record['songlength'] != 4*(i+1)
                or record['full_native_variables'] != 19+i
                or record['full_trainable_parameters'] != 4924855
                or not record['all_graph_nodes_and_library_bitwise_identical']):
            raise ValueError('Full original network, data, curriculum and optimizer budgets required')


def audit(config):
    from scripts.flow_matching.diagnose_crnngan_epoch_states_v1 import diagnose
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen three-way audit input changed')
    original_config = read(ROOT / config['original_diagnostic_configuration'])
    cache_config = read(ROOT / config['cache_diagnostic_configuration'])
    original_output = ROOT / original_config['output_directory']
    proof = read(original_output / 'original_four_epoch_repeatability.json')
    require_complete_original_diagnostic(proof, read(original_output / 'guard_receipt.json'))
    original_against_preserved = diagnose(original_config)
    cache_against_preserved = diagnose(cache_config)
    repeated_reference = dict(cache_config,
        reference_progress=str(Path(original_config['native_attempt_directory']) / 'progress.json'))
    cache_against_repeated_original = diagnose(repeated_reference)
    comparisons = dict(original_against_preserved=original_against_preserved,
                       cache_against_preserved=cache_against_preserved,
                       cache_against_repeated_original=cache_against_repeated_original)
    return dict(full_complete_epochs_each=4, comparisons=comparisons,
        original_execution_bitwise_repeatable_in_this_pair=original_against_preserved['all_trained_states_bitwise_identical'],
        cache_equivalence_gate_satisfied=cache_against_preserved['all_trained_states_bitwise_identical'],
        all_complete_native_graphs_bitwise_identical=True,
        no_cache_backend_adopted_by_this_diagnostic=True,
        full500_training_completed=False, benchmark_performance=False,
        author_equivalence_certified=False, all_original_experiments_complete=False,
        boundary='All three complete native attempts and every variable/graph/RNG/MIDI artifact read independently. '
                 'An original-executor repeatability difference does not pass the frozen cache-state equivalence gate. '
                 'This four-epoch diagnostic does not replace any of the35 full500-epoch jobs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_path']
    if target.exists():
        raise FileExistsError('Preserve every prior three-way full native diagnostic')
    result = audit(config)
    result['configuration_sha256'] = sha(ROOT / args.config)
    write(target, result)
    print('Three complete four-epoch native attempts audited. Original pair bitwise repeatable: %s; cache gate: %s' % (
        result['original_execution_bitwise_repeatable_in_this_pair'], result['cache_equivalence_gate_satisfied']))


if __name__ == '__main__':
    main()
