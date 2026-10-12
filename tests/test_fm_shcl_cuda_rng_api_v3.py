import ast
import inspect

from scripts.flow_matching import audit_shcl_cuda_rng_api_v3 as observed


def test_remove22_observations_recovers_every_original_CPU_CUDA_flag_and_restore_statement():
    tree = observed.helper_ast()
    class Remove(ast.NodeTransformer):
        count = 0
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'observe':
                self.count += 1
                return None
            return self.generic_visit(node)
    remove = Remove()
    tree = remove.visit(tree)
    assert remove.count == 22
    assert ast.dump(tree) == ast.dump(ast.parse(inspect.getsource(observed.stages.original.job_api.replay.restore_and_verify_draws)))


def test_suboperation_diagnosis_keeps_original_bounds_environment_and_full_task_scope():
    native = observed.stages.original.job_api.replay.original.native
    old = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v2.json')
    new = native.read(native.ROOT / 'configs/runtime/fm_shcl_native_CUDA_rng_API.v3.json')
    observed.stages.original.verify(new)
    for key in ('settings', 'gpu_settings', 'resource_lock', 'python', 'environment', 'original_scope',
            'full49_reference_configuration', 'diagnostic_only', 'canonical_training_jobs_credited'):
        assert new[key] == old[key]
    assert new['source_receipts'][:len(old['source_receipts'])] == old['source_receipts']
