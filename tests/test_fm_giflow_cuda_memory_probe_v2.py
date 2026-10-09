from scripts.flow_matching.giflow_cuda_memory_probe_v2 import bound_main
from scripts.flow_matching.run_giflow_cuda_memory_probe_v2 import bound_main as guarded_main
from scripts.flow_matching.run_giflow_cuda_memory_probe import main as previous_guard


def test_cuda_probe_uses_structural_lambda_validation():
    bound, changes = bound_main()
    assert all(value == 1 for value in changes.values())
    assert 'cuda_batch' in bound.__code__.co_consts
    assert 'integration_passed_not_benchmark' in bound.__code__.co_consts


def test_controller_only_changes_worker_module_binding():
    bound = guarded_main()
    pairs = [(a, b) for a, b in zip(previous_guard.__code__.co_consts, bound.__code__.co_consts) if a != b]
    assert pairs == [('scripts.flow_matching.giflow_cuda_memory_probe',
                      'scripts.flow_matching.giflow_cuda_memory_probe_v2')]
    assert bound.__globals__['guarded_gpu_wait'] is previous_guard.__globals__['guarded_gpu_wait']
