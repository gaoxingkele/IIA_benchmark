import ast
import inspect

from scripts.flow_matching import run_shcl_training_rng_queue_v2 as corrected
from scripts.flow_matching import run_shcl_training_rng_job_v1 as job


def test_exactly_two_unstarted_atomic_host_writes_corrected_all_other_code_preserved():
    function = corrected.compiled_main()
    original_tree = ast.parse(inspect.getsource(corrected.original.main))
    before_calls = [node for node in ast.walk(original_tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'native' and node.func.attr == 'write']
    assert len(before_calls) == 2
    original_tree = ast.fix_missing_locations(original_tree)
    for node in before_calls:
        node.func = ast.copy_location(ast.Name(id='write', ctx=ast.Load()), node.func)
    namespace = dict(corrected.original.__dict__, write=corrected.write)
    exec(compile(ast.fix_missing_locations(original_tree), corrected.__file__, 'exec'), namespace)
    assert function.__code__ == namespace['main'].__code__
    assert function.__globals__['write'] is corrected.write


def test_corrected_host_definition_keeps_every_original_task_policy_and_scientific_pin():
    native = job.replay.original.native
    old = native.read(native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v1.json')
    new = native.read(native.ROOT / 'configs/runtime/fm_shcl_all49_training_rng_replay.v2.json')
    job.verify(new)
    for key in ('training_job_ids', 'full_original_training_epochs', 'original_scope', 'settings', 'gpu_settings',
            'resource_lock', 'python', 'every_epoch_full_RNG_backend_restore_required', 'new_optimizer_updates_allowed'):
        assert new[key] == old[key]
    assert new['source_receipts'][:len(old['source_receipts'])] == old['source_receipts']
    assert not (native.ROOT / old['fixture_output_root']).exists()
