from scripts.flow_matching import audit_shcl_cuda_rng_api_v1 as audit


def test_actual_native_CUDA_abort_preserved_without_completion_or_performance_credit():
    native = audit.job_api.replay.original.native
    config = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v1.json')
    audit.verify(config)
    receipt = native.read(native.ROOT / config['state_root'] / 'receipt.json')
    status = native.read(native.ROOT / config['state_root'] / 'status.json')
    assert receipt['passed'] is False and receipt['resource_guard_aborted'] is True
    assert receipt['peak_tree_private_bytes'] > config['settings']['maximum_worker_private_bytes']
    assert receipt['peak_tree_rss_bytes'] < config['settings']['maximum_worker_rss_bytes']
    assert receipt['abort_snapshot']['commit_available_bytes'] > config['settings']['emergency_commit_headroom_bytes']
    assert status['state'] == 'failed_native_CUDA_rng_API_audit_preserved' and status['active_pid'] is None
    assert not (native.ROOT / config['output_root'] / 'actual_native_CUDA_rng_api_audit.json').exists()
    assert config['canonical_training_jobs_credited'] == 0
    # These checks validate truthful failure handling. They do not certify the
    # native CUDA restoration API or any canonical checkpoint restoration.
