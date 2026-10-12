import ast
import inspect

from scripts.flow_matching import audit_shcl_cuda_rng_api_v2 as observed


def test_observation_removal_recovers_every_original_native_computation_and_guard():
    tree = observed.worker_ast()
    calls = [node for node in tree.body[0].body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'observe']
    assert len(calls) == 7
    tree.body[0].body = [node for node in tree.body[0].body if node not in calls]
    assert ast.dump(tree) == ast.dump(ast.parse(inspect.getsource(observed.original.worker)))


def test_new_observation_registration_preserves_old_policy_scope_environment_and_API_source():
    native = observed.original.job_api.replay.original.native
    old = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v1.json')
    new = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v2.json')
    observed.original.verify(new)
    for key in ('settings', 'gpu_settings', 'resource_lock', 'python', 'environment', 'original_scope',
            'full49_reference_configuration', 'diagnostic_only', 'canonical_training_jobs_credited'):
        assert new[key] == old[key]
    assert new['source_receipts'][:len(old['source_receipts'])] == old['source_receipts']
