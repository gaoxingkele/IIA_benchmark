import copy

import pytest

from scripts.flow_matching import audit_shcl_cuda_rng_api_v1 as audit


def config():
    native = audit.job_api.replay.original.native
    return native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v1.json')


def test_actual_CUDA_API_diagnostic_keeps_every_original_guard_and_full49_source_gate():
    cfg = config()
    old = audit.verify(cfg)
    assert len(old['training_job_ids']) == 49 and old['full_original_training_epochs'] == 155
    assert old['full_original_source_commands'] == 413
    assert cfg['canonical_training_jobs_credited'] == 0 and cfg['diagnostic_only']
    assert cfg['gpu_settings']['minimum_gpu_free_bytes'] == 8 * 2**30
    assert cfg['gpu_settings']['emergency_gpu_free_bytes'] == 2 * 2**30


@pytest.mark.parametrize('mutation', ['physical', 'commit', 'private', 'GPU', 'credit', 'diagnostic'])
def test_diagnostic_cannot_weaken_original_safeguards_or_satisfy_training(mutation):
    cfg = copy.deepcopy(config())
    if mutation == 'physical':
        cfg['settings']['minimum_available_memory_bytes'] -= 1
    elif mutation == 'commit':
        cfg['settings']['minimum_commit_headroom_bytes'] -= 1
    elif mutation == 'private':
        cfg['settings']['maximum_worker_private_bytes'] += 1
    elif mutation == 'GPU':
        cfg['gpu_settings']['minimum_gpu_free_bytes'] -= 1
    elif mutation == 'credit':
        cfg['canonical_training_jobs_credited'] = 1
    else:
        cfg['diagnostic_only'] = False
    with pytest.raises(ValueError):
        audit.verify(cfg)
