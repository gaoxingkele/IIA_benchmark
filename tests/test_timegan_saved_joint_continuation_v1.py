"""Real author AST fidelity and fail-closed saved-state continuation gates."""
import ast
import copy
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.flow_matching import timegan_saved_joint_continuation_v1 as continuation


CONFIG = 'configs/reproducibility/fm_timegan_sine_saved_joint15000_continuation.2026-10-14_v1.json'


def test_real_author_joint_body_and_post_training_unchanged():
    config = continuation.read(continuation.ROOT / CONFIG)
    queue, model, job, saved, audit = continuation.registration(config)
    module = SimpleNamespace(__file__=str(continuation.ROOT / model['source_root'] / 'timegan.py'))
    original = continuation.stages.resumed_ast(module).body[0]
    actual = continuation.joint_ast(module).body[0]
    restored = copy.deepcopy(actual)
    loop = next(n for n in restored.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'itt')
    assert ast.dump(loop.iter) == ast.dump(ast.parse('range(15000, iterations)', mode='eval').body)
    assert len(range(15000, 50000)) == 35000
    assert ast.dump(loop.body[-1]) == ast.dump(ast.parse('_save_complete_joint_boundary(sess, itt, locals())').body[0])
    loop.body.pop(); loop.iter = ast.parse('range(iterations)', mode='eval').body
    assert ast.dump(restored) == ast.dump(original)
    assert len(queue['jobs']) == 45
    assert saved['optimizer_updates'] == dict(E0=50000, GS=50000, E=30000, G=30000, D=15000)


def test_compiler_changes_only_state_path_and_prior_counts():
    actual = continuation.compiler_tree()
    original = continuation.observer.compiler_ast(inspect.getsource(continuation.stages.compile_observed_resume))
    restored = copy.deepcopy(actual)
    original_assigns = {n.targets[0].id: n for n in ast.walk(original) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('prior', 'directory')}
    for node in ast.walk(restored):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in original_assigns:
            node.value = copy.deepcopy(original_assigns[node.targets[0].id].value)
    assert ast.dump(restored) == ast.dump(original)


def test_missing_actual_native_audit_cannot_compile_training():
    config = continuation.read(continuation.ROOT / CONFIG)
    with patch.object(continuation, 'require_native_proof', side_effect=FileNotFoundError('audit absent')):
        with pytest.raises(FileNotFoundError):
            continuation.compile_continuation(config, '0' * 64)


@pytest.mark.parametrize('field,value', [('remaining_joint_iterations', 34999),
    ('joint_iterations', 49999), ('independent_training_seed_slots_added', 1)])
def test_no_short_budget_or_extra_seed(field, value):
    config = continuation.read(continuation.ROOT / CONFIG)
    config[field] = value
    with pytest.raises(ValueError, match='Original full scope'):
        continuation.registration(config)


def test_original_guard_and_admission_preserved():
    config = continuation.read(continuation.ROOT / CONFIG)
    queue, _, _, _, audit = continuation.registration(config)
    assert config['settings'] == dict(queue['settings'], minimum_commit_headroom_bytes=40 * 1024**3)
    assert audit['resource_lock'] == queue['resource_lock']
    assert config['original_training_settings'] == queue['settings']


def test_actual_proof_must_be_passed_guarded_and_complete():
    config = continuation.read(continuation.ROOT / CONFIG)
    _, model, job, saved, audit = continuation.registration(config)
    proof = dict(native_restore=dict(restored_native_variables=284,
        all_variable_values_bitwise_equal=True, all_variables_finite=True,
        stochastic_training_or_generation_ancestors=[], checkpoint_index_sha256='a' * 64,
        rng_sha256='a' * 64), passed=True, canonical_job_id=job['id'],
        saved_optimizer_updates=saved['optimizer_updates'], completed_joint_iterations=15000,
        remaining_original_joint_iterations=35000, required_original_joint_iterations=50000,
        full_native_parameters=saved['model_parameters'], full_original_data_shape=[10000, 24, 5],
        native_NumPy_and_Python_RNG_bitwise_restored=True,
        new_optimizer_updates_executed=0, additional_training_seed_slots=0)
    guard = dict(exit_code=0, resource_guard_aborted=False, full_native_joint15000_restore_passed=True)
    for mutation in ['passed', 'variable_count', 'guard_abort', 'digest', 'rng', 'training']:
        p, g = copy.deepcopy(proof), copy.deepcopy(guard)
        expected = 'a' * 64
        if mutation == 'passed': p['passed'] = False
        if mutation == 'variable_count': p['native_restore']['restored_native_variables'] = 283
        if mutation == 'guard_abort': g['resource_guard_aborted'] = True
        if mutation == 'digest': expected = 'b' * 64
        if mutation == 'rng': p['native_NumPy_and_Python_RNG_bitwise_restored'] = False
        if mutation == 'training': p['new_optimizer_updates_executed'] = 1
        with patch.object(continuation, 'registration', return_value=({}, model, job, saved, audit)), \
             patch.object(continuation, 'read', side_effect=[g, p, {'datasets': {'sine': {'full_shape': [10000, 24, 5]}}}]), \
             patch.object(continuation, 'sha', return_value='a' * 64):
            with pytest.raises(ValueError, match='Actual guarded'):
                continuation.require_native_proof(config, expected)


def test_real_full_observer_compiles_without_importing_training_framework():
    config = continuation.read(continuation.ROOT / CONFIG)
    _, _, job, saved, audit = continuation.registration(config)
    with patch.object(continuation, 'require_native_proof', return_value=(saved, audit, 'a' * 64)):
        run, queue, recovered_job = continuation.compile_continuation(config, 'a' * 64)
    assert callable(run)
    assert len(queue['jobs']) == 45
    assert recovered_job == dict(job, output_directory=config['output_directory'])
