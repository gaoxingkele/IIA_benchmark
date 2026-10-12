import copy

import pytest

from scripts.flow_matching import run_shcl_training_rng_job_v1 as job
from scripts.flow_matching import run_shcl_training_rng_queue_v1 as queue
from scripts.flow_matching import shcl_training_rng_replay_v1 as replay


def config():
    return replay.original.native.read(replay.original.native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v1.json')


def test_all49_original_training_jobs_and155_epochs_plus_full413_obligations_retained():
    cfg = config()
    original, model = job.verify(cfg)
    jobs = [item for item in original['jobs'] if item['mode'] == 'train']
    assert len(jobs) == 49 and sum(item['num_epochs'] for item in jobs) == 155
    assert cfg['training_job_ids'] == [item['id'] for item in jobs]
    assert cfg['full_original_source_commands'] == 413 and cfg['new_optimizer_updates_allowed'] == 0


@pytest.mark.parametrize('mutation', ['jobs', 'physical', 'commit', 'private', 'GPU', 'updates', 'scope'])
def test_cannot_shorten_original_selection_or_weaken_existing_frozen_guards(mutation):
    cfg = copy.deepcopy(config())
    if mutation == 'jobs':
        cfg['training_job_ids'].pop()
    elif mutation == 'physical':
        cfg['settings']['minimum_available_memory_bytes'] -= 1
    elif mutation == 'commit':
        cfg['settings']['minimum_commit_headroom_bytes'] -= 1
    elif mutation == 'private':
        cfg['settings']['maximum_worker_private_bytes'] += 1
    elif mutation == 'GPU':
        cfg['gpu_settings']['minimum_gpu_free_bytes'] -= 1
    elif mutation == 'updates':
        cfg['new_optimizer_updates_allowed'] = 1
    else:
        cfg['original_scope']['method_baseline_ablation_records'] -= 1
    with pytest.raises(ValueError):
        job.verify(cfg)


@pytest.mark.parametrize('mutation', ['rng_missing', 'flag_missing', 'flag_numeric'])
def test_missing_saved_training_state_fails_instead_of_defaulting(mutation):
    checkpoint = dict(original_rng={key: None for key in ('python', 'numpy', 'torch_cpu', 'torch_cuda')},
        backend_flags={key: True for key in ('cudnn_benchmark', 'cudnn_deterministic', 'cudnn_enabled',
            'cudnn_allow_tf32', 'matmul_allow_tf32', 'deterministic_algorithms')})
    if mutation == 'rng_missing':
        del checkpoint['original_rng']['numpy']
    elif mutation == 'flag_missing':
        del checkpoint['backend_flags']['cudnn_enabled']
    else:
        checkpoint['backend_flags']['cudnn_enabled'] = 1
    with pytest.raises(ValueError):
        replay.verify_schema(checkpoint)


def test_fixture_never_satisfies_real_training_or_unverified_CUDA_state():
    source = dict(num_epochs=1)
    row = dict(all_original_training_RNG_states_bitwise_restored=True,
        all_original_training_backend_flags_exactly_restored=True, independent_native_stream_draws_bitwise_verified=True,
        original_visible_CUDA_generators_verified=0)
    evidence = dict(passed=True, source_training_job=source, every_original_training_RNG_and_backend_state_verified=True,
        all_original_training_epoch_model_Adam_RNG_backend_restore_verified=True,
        synthetic_operator_only_not_benchmark_performance=True, new_optimizer_updates_executed=0,
        new_benchmark_results=0, training_rng_checkpoints=[row])
    assert queue.accepted(evidence, source, True)
    assert not queue.accepted(evidence, source, False)
    evidence['synthetic_operator_only_not_benchmark_performance'] = False
    assert not queue.accepted(evidence, source, False)
    row['original_visible_CUDA_generators_verified'] = 1
    assert queue.accepted(evidence, source, False)
    row['independent_native_stream_draws_bitwise_verified'] = False
    assert not queue.accepted(evidence, source, False)
