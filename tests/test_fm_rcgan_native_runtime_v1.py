"""Full-source invariants and explicit native numerical bridge failure gates."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.rcgan_mmd_runtime_v1 import kernel_ast, repair_proven_zero_distances
from scripts.flow_matching.rcgan_native_training_runtime_v1 import experiment_ast
from scripts.flow_matching.run_rcgan_native_original_queue_v1 import verify_registration
from scripts.flow_matching.audit_rcgan_mmd_native_v4 import retain_original_fit


def source_root():
    registry = read(ROOT / 'configs/acquisition/fm_adopted_gan_original_sources.v1.json')
    return Path(next(a['original_directory'] for a in registry['assets'] if a['id'] == 'rcgan'))


def test_kernel_formula_preserved_except_exact_gram_bindings():
    module = SimpleNamespace(__file__=str(source_root() / 'mmd.py'))
    original = next(n for n in ast.parse(Path(module.__file__).read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_mix_rbf_kernel')
    restored = kernel_ast(module)
    restored.name = original.name
    originals = [n for n in ast.walk(original) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ('XX', 'XY', 'YY')]
    modified = [n for n in ast.walk(restored) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ('XX', 'XY', 'YY')]
    assert len(originals) == len(modified) == 6
    for new, old in zip(modified, originals):
        new.value = copy.deepcopy(old.value)
    assert ast.dump(restored) == ast.dump(original)


def test_original_full_experiment_loop_and_other_statements_unchanged():
    path = source_root() / 'experiment.py'
    original = ast.parse(path.read_text(encoding='utf-8'))
    restored = experiment_ast(path)
    epoch = next(n for n in restored.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    hook = epoch.body.pop()
    assert isinstance(hook, ast.Expr) and hook.value.func.id == '_native_epoch_complete'
    index = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Import) and [a.name for a in n.names] == ['eval'])
    restored.body.insert(index, copy.deepcopy(original.body[index]))
    assert ast.dump(restored) == ast.dump(original)


def test_roundoff_repairs_only_identical_vectors_and_preserves_inputs():
    data = np.array([[1., 2.], [3., 4.]])
    before = data.copy()
    squared = np.array([[-1e-12, 8.], [8., -1e-12]])
    evidence = repair_proven_zero_distances(squared, data, data)
    assert evidence['negative_pairs'] == 2
    assert np.array_equal(squared, [[0., 8.], [8., 0.]])
    assert np.array_equal(data, before)


def test_nonidentical_negative_distance_is_not_hidden():
    data = np.array([[1., 2.], [3., 4.]])
    with pytest.raises(ValueError, match='nonidentical'):
        repair_proven_zero_distances(np.array([[0., -1e-12], [8., 0.]]), data, data)


def test_nonfinite_distance_is_not_hidden():
    with pytest.raises(ValueError, match='Nonfinite'):
        repair_proven_zero_distances(np.array([[np.nan]]), np.array([[1.]]), np.array([[1.]]))


def test_exact_original_source_epoch_update_budget():
    function = next(n for n in ast.parse((source_root() / 'model.py').read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    loop = next(n for n in function.body if isinstance(n, ast.For) and n.target.id == 'batch_idx')
    values = eval(compile(ast.Expression(loop.iter), '<original_full_batch_schedule>', 'eval'),
                  dict(samples=np.empty((36000, 0)), batch_size=28, D_rounds=1, G_rounds=1, cond_dim=0))
    assert len(values) == 1284
    assert 500 * len(values) == 642000


def test_no_missing_seed_or_smaller_epoch_budget_allowed():
    queue = dict(source_receipts=[], jobs=[dict(seed=s, id='rcgan_mnistfull__seed'+str(s), output_directory=str(s))
                 for s in range(1103, 1108)], epochs=500, D_updates_per_epoch=1284, G_updates_per_epoch=1284,
                 sigma_maximum_updates_per_evaluation=2000, full_evaluations_per_job=500)
    verify_registration(queue)
    with pytest.raises(ValueError, match='five'):
        verify_registration(dict(queue, jobs=queue['jobs'][:-1]))
    with pytest.raises(ValueError, match='budgets'):
        verify_registration(dict(queue, epochs=1))


def test_native_fit_optimizer_not_replaced_by_rejected_cached_gradient():
    path = ROOT / 'scripts/flow_matching/audit_rcgan_mmd_native_v1.py'
    function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    revised = retain_original_fit(function)
    text = ast.unparse(revised)
    assert "minimize(-cases['fit']['original'][1]" in text
    assert "[sigma, cases['fit']['original'][1], sigma_solver]" in text
    assert "sigma_iter < 2000" in text and 'that_change > 0.001' in text
    assert 'allclose(literal_values, original_test' in text


def test_training_worker_returns_unchanged_original_fit_function():
    tree = ast.parse((ROOT / 'scripts/flow_matching/rcgan_native_training_runtime_v1.py').read_text(encoding='utf-8'))
    mixture = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'mixture')
    original_fit = next(n for n in ast.walk(mixture) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == 'fit_value')
    assert isinstance(original_fit.value, ast.Call) and original_fit.value.func.id == 'original_mixture'
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id == 'fit_value' for n in ast.walk(mixture))
    assert 'cached_kernel' not in ast.unparse(mixture)
