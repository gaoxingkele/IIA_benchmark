"""Add complete saved training RNG restoration to all49 original training jobs."""
import argparse
from pathlib import Path

from scripts.flow_matching import shcl_training_rng_replay_v1 as replay
from scripts.flow_matching.run_shcl_native_replay_job_v1 import verify as verify_previous
from scripts.flow_matching.run_shcl_native_script_queue_v1 import complete
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def verify(config):
    native = replay.original.native
    for pin in config['source_receipts']:
        if native.sha(native.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen additive training RNG replay changed: ' + pin['path'])
    old = native.read(native.ROOT / config['original_full413_replay_configuration'])
    queue, model = verify_previous(old)
    jobs = [job for job in queue['jobs'] if job['mode'] == 'train']
    if config['training_job_ids'] != [job['id'] for job in jobs] or len(jobs) != 49:
        raise ValueError('All49 original source training commands required')
    if config['original_scope'] != queue['original_scope'] or config['full_original_source_commands'] != 413:
        raise ValueError('Keep the complete original training and test obligations')
    if (config['settings'] != old['settings'] or config['gpu_settings'] != old['gpu_settings']
            or config['resource_lock'] != old['resource_lock'] or config['python'] != old['python']):
        raise ValueError('All original read-only CPU/private/commit/GPU safeguards retained')
    if not config['every_epoch_full_RNG_backend_restore_required'] or config['new_optimizer_updates_allowed'] != 0:
        raise ValueError('Every original saved state required without training or tolerance relaxation')
    return queue, model


def worker(config, config_path, job_id=None, fixture=False):
    native = replay.original.native
    queue, model = verify(config)
    if replay.original.environment() != queue['environment']:
        raise ValueError('Same original native numerical environment required')
    import torch
    torch.set_num_threads(2)
    if fixture:
        reference = native.read(native.ROOT / config['saved_fixture_configuration'])
        training = native.read(reference['fixture_training'])
        checkpoint = Path(reference['fixture_checkpoint'])
        if not training['job'].get('synthetic_fixture'):
            raise ValueError('Actual saved diagnostic fixture required; no canonical credit')
        output = native.ROOT / config['fixture_output_root']
    else:
        job = next(job for job in queue['jobs'] if job['id'] == job_id and job['mode'] == 'train')
        if not complete(job):
            raise ValueError('Full original-data native training required before checkpoint audit')
        training = native.read(Path(job['output_directory']) / 'observer/result.json')
        checkpoint = Path(job['checkpoint_directory']) / (job['dataset'] + '_checkpoint.pth')
        output = Path(config['full_training_output_root']) / job_id
    if output.exists():
        raise FileExistsError('Preserve every complete or failed independent RNG replay attempt')
    output.mkdir(parents=True)
    report = dict(passed=False, configuration=config_path, configuration_sha256=native.sha(native.ROOT / config_path),
        source_training_job=training['job'], original_scope=queue['original_scope'],
        full_original_training_commands=49, full_original_test_commands=364, canonical_training_credit=False,
        all_full413_original_source_replay_obligations_retained=True)
    replay.replay_training(model, training, checkpoint, report)
    report['passed'] = True
    exclusive(output / 'full_checkpoint_rng_restore.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--fixture', action='store_true')
    selection.add_argument('--job-id')
    args = parser.parse_args()
    worker(replay.original.native.read(replay.original.native.ROOT / args.config), args.config, args.job_id, args.fixture)


if __name__ == '__main__':
    main()
