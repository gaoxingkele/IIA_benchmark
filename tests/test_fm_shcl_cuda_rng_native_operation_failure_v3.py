from scripts.flow_matching import audit_shcl_cuda_rng_api_v3 as observed


def test_actual_failure_boundary_is_first_CUDA_draw_after_successful_state_restore_readback():
    native = observed.stages.original.job_api.replay.original.native
    config = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v3.json')
    observed.stages.original.verify(config)
    memory = native.read(native.ROOT / config['state_root'] / 'native_operation_memory.json')
    receipt = native.read(native.ROOT / config['state_root'] / 'receipt.json')
    phases = [row['phase'] for row in memory['observations']]
    assert 'after_native_CUDA_RNG_initialization' in phases
    assert 'after_native_backend_flag_restore' in phases and 'after_native_saved_RNG_readback' in phases
    assert 'after_native_CUDA_generator_state_restore' in phases
    assert phases[-1] == 'before_independent_native_CUDA_draw'
    assert 'after_independent_native_CUDA_draw' not in phases
    assert receipt['resource_guard_aborted'] and not receipt['passed']
    assert receipt['peak_tree_private_bytes'] > config['settings']['maximum_worker_private_bytes']
    assert all(row['private_bytes'] < config['settings']['maximum_worker_private_bytes'] for row in memory['observations'])
    assert not (native.ROOT / config['output_root'] / 'actual_native_CUDA_rng_api_audit.json').exists()
    assert memory['canonical_training_credit'] == 0


def test_prior_stage_diagnostic_and_original_failed_attempt_remain_failed_and_preserved():
    native = observed.stages.original.job_api.replay.original.native
    for version in (1, 2):
        config = native.read(native.ROOT / ('configs/runtime/fm_shcl_native_CUDA_rng_API.v' + str(version) + '.json'))
        observed.stages.original.verify(config)
        receipt = native.read(native.ROOT / config['state_root'] / 'receipt.json')
        assert not receipt['passed'] and receipt['resource_guard_aborted']
        assert config['canonical_training_jobs_credited'] == 0
