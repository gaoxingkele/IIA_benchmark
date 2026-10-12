from scripts.flow_matching import run_shcl_training_rng_job_v1 as job
from scripts.flow_matching import run_shcl_training_rng_queue_v1 as queue


def test_actual_native_saved_full_state_restoration_passes_without_benchmark_credit():
    native = job.replay.original.native
    config = native.read(native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v2.json')
    report = native.read(native.ROOT / config['fixture_output_root'] / 'full_checkpoint_rng_restore.json')
    receipt = native.read(native.ROOT / config['fixture_state_root'] / 'receipts/saved_full_native_fixture.json')
    source = native.read(native.ROOT / config['saved_fixture_configuration'])
    training = native.read(source['fixture_training'])
    assert queue.accepted(report, training['job'], True)
    assert not queue.accepted(report, training['job'], False)
    assert receipt['passed'] and receipt['exit_code'] == 0 and not receipt['resource_guard_aborted']
    assert receipt['peak_tree_rss_bytes'] <= config['settings']['maximum_worker_rss_bytes']
    assert receipt['peak_tree_private_bytes'] <= config['settings']['maximum_worker_private_bytes']
    assert report['canonical_training_credit'] is False
    assert report['new_optimizer_updates_executed'] == 0 and report['new_benchmark_results'] == 0
    assert report['full_original_training_commands'] == 49 and report['full_original_test_commands'] == 364
    assert report['all_full413_original_source_replay_obligations_retained']
    assert report['training_checkpoints'][0]['all_model_and_Adam_state_bitwise_restored']
    assert report['training_rng_checkpoints'][0]['original_visible_CUDA_generators_verified'] == 0
    assert not report['original_paper_equivalence_certified'] and not report['all_original_experiments_complete']
