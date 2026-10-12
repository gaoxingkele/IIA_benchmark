"""Preserve the unstarted v1 definition and register corrected host IO bindings."""
import copy
import json

from scripts.flow_matching import run_shcl_training_rng_job_v1 as job
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    native = job.replay.original.native
    previous_path = 'configs/runtime/fm_shcl_all49_training_rng_replay.v1.json'
    previous = native.read(native.ROOT / previous_path)
    job.verify(previous)
    if any((native.ROOT / previous[key]).exists() for key in ('state_root', 'fixture_state_root', 'fixture_output_root')):
        raise ValueError('Unstarted v1 definition must be preserved, never hot edited')
    config = copy.deepcopy(previous)
    for key in ('state_root', 'fixture_state_root', 'fixture_output_root', 'full_training_output_root'):
        config[key] = config[key].replace('_v1', '_v2')
    extra = [previous_path, 'scripts/flow_matching/run_shcl_training_rng_queue_v2.py',
        'scripts/flow_matching/register_shcl_training_rng_replay_v2.py', 'tests/test_fm_shcl_training_rng_queue_v2.py']
    config['source_receipts'] += [dict(path=path, sha256=native.sha(native.ROOT / path)) for path in extra]
    config['execution_only_correction'] = 'Bind exactly two unstarted host status/receipt calls to the repository atomic writer. No native source, RNG check, guard, budget, dataset, task or criterion changes.'
    config['unstarted_v1_preserved'] = previous_path
    job.verify(config)
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v2.json', config)
    print(json.dumps(dict(original_training_commands=49, full_original_epochs=155,
        full_source_commands_retained=413, original_optimizer_updates_added=0)))


if __name__ == '__main__':
    main()
