"""Freeze all49 additive saved-state RNG gates; retain existing413 obligations."""
import json

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    old_path = 'configs/runtime/fm_shcl_full413_completion_replay.v1.json'
    old = native.read(native.ROOT / old_path)
    queue = native.read(native.ROOT / old['original_queue'])
    fixture_path = 'configs/runtime/fm_shcl_independent_replay_fixture.v1.json'
    fixture = native.read(native.ROOT / fixture_path)
    training = native.read(fixture['fixture_training'])
    paths = [old_path, old['original_queue'], fixture_path,
        'scripts/flow_matching/shcl_training_rng_replay_v1.py', 'scripts/flow_matching/run_shcl_training_rng_job_v1.py',
        'scripts/flow_matching/run_shcl_training_rng_queue_v1.py', 'scripts/flow_matching/register_shcl_training_rng_replay_v1.py',
        'tests/test_fm_shcl_training_rng_replay_v1.py', 'scripts/flow_matching/shcl_native_observer_v1.py',
        'scripts/flow_matching/replay_shcl_native_frames_v1.py', 'scripts/flow_matching/run_shcl_native_replay_job_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py', 'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    paths += [fixture['fixture_training'], fixture['fixture_checkpoint']] + [row['full_state'] for row in training['epochs']]
    config = dict(original_full413_replay_configuration=old_path, saved_fixture_configuration=fixture_path,
        original_scope=queue['original_scope'], full_original_source_commands=413,
        training_job_ids=[job['id'] for job in queue['jobs'] if job['mode'] == 'train'],
        full_original_training_epochs=sum(job['num_epochs'] for job in queue['jobs'] if job['mode'] == 'train'),
        settings=old['settings'], gpu_settings=old['gpu_settings'], resource_lock=old['resource_lock'],
        python=old['python'], environment=old['environment'],
        state_root='experiments/runs/fm_shcl_all49_training_rng_replay_v1',
        fixture_state_root='experiments/runs/fm_shcl_saved_training_rng_fixture_v1',
        fixture_output_root='projects/flow_matching_research/results/2026-10-12/shcl_training_rng_restoration_v1/fixture',
        full_training_output_root='F:/aicoding/IIA_Data/experiments/shcl_released_native_v1/independent_training_rng_replays_v1',
        every_epoch_full_RNG_backend_restore_required=True, new_optimizer_updates_allowed=0,
        source_receipts=[dict(path=p, sha256=native.sha(native.ROOT / p)) for p in paths],
        boundary='Additive gate for every saved epoch from all49 complete original trainings (155 epochs). '
            'Restore full Python/NumPy/native CPU/all original CUDA streams and six backend flags, verify '
            'independent native draws bitwise, then restore saved and caller states. Existing full413 '
            'source replay and strict grouped full-point TAB obligations remain. The saved diagnostic '
            'fixture verifies CPU restoration only and never provides training or performance credit.')
    exclusive(native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v1.json', config)
    print(json.dumps(dict(original_training_commands=len(config['training_job_ids']), full_original_epochs=config['full_original_training_epochs'],
        full_source_commands_retained=413, original_optimizer_updates_added=0)))


if __name__ == '__main__':
    main()
