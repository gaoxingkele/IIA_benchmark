"""Gate all35 unchanged full native jobs on real four-epoch state/RNG evidence."""
import copy
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.run_crnngan_native_original_queue_v2 import verify_registration
from scripts.flow_matching.register_crnngan_full_memory_recovery_v3 import verify_scientific_equivalence

AUDIT = 'projects/flow_matching_research/results/2026-10-11/crnngan_four_complete_epochs_v1/four_complete_epoch_executor_audit.json'


def registration(previous, original, audit):
    if (audit['full_complete_epochs_verified'] != 4 or not audit['all_full_native_train_states_bitwise_identical']
            or not audit['all_original_graph_geometry_and_random_seeds_identical']
            or len(audit['records']) != 4
            or [r['epoch'] for r in audit['records']] != list(range(4))
            or any(not r['all_native_variable_hashes_bitwise_identical']
                   or not r['whole_graph_geometry_and_rng_seeds_identical']
                   or r['trainable_parameters'] != 4924855
                   or r['full_native_variables'] != 19 + i
                   or r['full_training_batches'] != 130
                   or r['songlength'] != 4 * (i + 1)
                   or r['optimizer_updates'] != dict(pretraining=130 * (i + 1), G=0, D=0)
                   for i, r in enumerate(audit['records']))):
        raise ValueError('All four real complete native epoch states and original RNG graphs must pass')
    queue = copy.deepcopy(previous)
    queue['id'] = 'crnngan_classical_released_full_native_executor_cache_v4'
    queue['state_root'] = 'experiments/runs/fm_crnngan_native_classical_released_v4'
    queue['prior_attempt_queue'] = 'configs/experiments/fm_crnngan_native_classical_released_queue.v3.json'
    queue['settings'] = copy.deepcopy(original['settings'])
    for job in queue['jobs']:
        job['output_directory'] = 'F:/aicoding/IIA_Data/experiments/crnngan_native_classical_released_v4/jobs/' + job['variant'] + '/seed' + str(job['seed'])
    for path in [AUDIT, 'scripts/flow_matching/crnngan_executor_cache_runtime_v1.py',
                 'scripts/flow_matching/run_crnngan_executor_cache_queue_v1.py',
                 'scripts/flow_matching/register_crnngan_executor_cache_v4.py',
                 'tests/test_fm_crnngan_executor_cache_v1.py']:
        queue['source_receipts'].append(dict(path=path, sha256=sha(ROOT/path)))
    queue['recovery_boundary'] = ('All35 canonical method/seed jobs and full500 epochs/full350x2/batch20/curriculum100/original3014-MIDI heldouts unchanged. '
        'Retain every original TF graph node and random-operation seed; normalize dependency-free inert fetches only. '
        'Four whole original source epochs/native trained states and RNG metadata verified bitwise; raw scalar loss differences reported rather than certified identical. '
        'Original24GiB physical/28GiB commit admission and10GiB RSS/12GiB private/16GiB emergency guards restored for this bounded-executor backend. '
        'Full100 curriculum capacity and complete500-epoch outcomes still require actual verification; failed attempts remain preserved, with no extra independent recovery seeds.')
    verify_scientific_equivalence(previous, queue)
    verify_registration(queue)
    return queue


def main():
    previous = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v3.json')
    original = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    graphs = read(ROOT/'projects/flow_matching_research/results/2026-10-11/crnngan_four_complete_epochs_v1/independent_all_graph_nodes.json')
    if graphs['full_original_epochs'] != 4 or not graphs['all_complete_native_graph_definitions_bitwise_identical']:
        raise ValueError('Independent complete graph node definitions must pass')
    queue = registration(previous, original, read(ROOT/AUDIT))
    for path in ['projects/flow_matching_research/results/2026-10-11/crnngan_four_complete_epochs_v1/independent_all_graph_nodes.json',
                 'scripts/flow_matching/audit_crnngan_complete_graph_nodes_v1.py']:
        queue['source_receipts'].append(dict(path=path, sha256=sha(ROOT/path)))
    receipt = read(ROOT/'projects/flow_matching_research/results/2026-10-11/crnngan_four_complete_epochs_v1/guard_receipt.json')
    if receipt['exit_code'] != 0 or not receipt['passed'] or receipt['resource_guard_aborted']:
        raise ValueError('Successful terminal real native audit required')
    path = ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v4.json'
    if path.exists() or any(Path(j['output_directory']).exists() for j in queue['jobs']):
        raise FileExistsError('Preserve every prior full native attempt and registration')
    write(path, queue)


if __name__ == '__main__':
    main()
