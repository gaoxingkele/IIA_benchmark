import json
import pytest
from scripts.flow_matching.run_giflow_serial_native_queue_v3 import bound_main, memory_preflight_ready
from scripts.flow_matching.run_giflow_guarded_gpu_queue_v2 import bound_main as previous


def test_serial_native_dispatch_retains_child_and_all_emergency_guards():
    bound, old = bound_main(), previous()
    changes = [(a, b) for a, b in zip(old.__code__.co_consts, bound.__code__.co_consts) if a != b]
    assert changes == [('configs/experiments/fm_giflow_native_queue.v2.json',
                        'configs/experiments/fm_giflow_native_queue.v3.json')]
    assert bound.__code__.co_code == old.__code__.co_code
    assert bound.__globals__['guarded_gpu_wait'] is old.__globals__['guarded_gpu_wait']
    assert 'scripts.flow_matching.run_giflow_native_job' in bound.__code__.co_consts


def test_memory_preflight_is_serial_and_is_never_promoted_to_formal_metrics():
    from scripts.flow_matching.run_giflow_cuda_memory_probe_v3 import bound_main as probe
    from scripts.flow_matching.run_giflow_cuda_memory_probe import guarded_gpu_wait
    bound = probe()
    assert 'scripts.flow_matching.giflow_cuda_memory_probe_v2' in bound.__code__.co_consts
    assert bound.__globals__['guarded_gpu_wait'] is guarded_gpu_wait
    assert 'cuda_full_batch_memory_preflight_passed_not_benchmark' in bound.__code__.co_consts


def test_both_profiles_and_sufficient_measured_headroom_are_required(tmp_path, monkeypatch):
    from scripts.flow_matching.giflow_native_protocol import sha
    import scripts.flow_matching.giflow_cuda_memory_probe as module
    monkeypatch.setattr(module, 'verify_config', lambda config, root: None)
    config = {'resource_lock': 'shared.lock', 'state_root': 'state',
              'profiles': [{'id': track, 'output_directory': track} for track in ['mirror', 'corrected']]}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'state').mkdir()
    queue = {'memory_preflight_config': 'config.json', 'memory_preflight_config_sha256': sha(tmp_path / 'config.json'),
             'resource_lock': 'shared.lock', 'settings': {'minimum_commit_headroom_bytes': 25 * 1024**3,
             'emergency_commit_headroom_bytes': 12 * 1024**3, 'minimum_gpu_free_bytes': 20 * 1024**3,
             'emergency_gpu_free_bytes': 4 * 1024**3}}
    assert not memory_preflight_ready(queue, tmp_path)
    for profile in config['profiles']:
        output = tmp_path / profile['output_directory']; output.mkdir()
        (output / 'integration_result.json').write_text('{"diagnostic":true}')
        result = {'status': 'cuda_full_batch_memory_preflight_passed_not_benchmark',
                  'probe_config_sha256': sha(tmp_path / 'config.json'), 'formal_metrics_published': False,
                  'full_registered_batch_size': 128, 'optimizer_updates': 1,
                  'integration_result_sha256': sha(output / 'integration_result.json'),
                  'peak_cuda_reserved_bytes': 10 * 1024**3}
        (output / 'cuda_memory_result.json').write_text(json.dumps(result))
        usage = {'exit_code': 0, 'resource_guard_aborted': False, 'peak_tree_private_bytes': 11 * 1024**3,
                 'result_sha256': sha(output / 'cuda_memory_result.json')}
        (tmp_path / 'state' / (profile['id'] + '.resource_usage.json')).write_text(json.dumps(usage))
        if profile['id'] == 'mirror':
            assert not memory_preflight_ready(queue, tmp_path)
    assert memory_preflight_ready(queue, tmp_path)
    queue['settings']['minimum_commit_headroom_bytes'] = 16 * 1024**3
    assert not memory_preflight_ready(queue, tmp_path)
    queue['settings']['minimum_commit_headroom_bytes'] = 25 * 1024**3
    usage['resource_guard_aborted'] = True
    (tmp_path / 'state' / 'corrected.resource_usage.json').write_text(json.dumps(usage))
    with pytest.raises(ValueError, match='evidence is incomplete'):
        memory_preflight_ready(queue, tmp_path)
