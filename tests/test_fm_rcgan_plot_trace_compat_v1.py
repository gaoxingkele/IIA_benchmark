"""Ensure trace compatibility cannot change original training or missing values."""
import ast
import copy
from pathlib import Path
import pytest

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.rcgan_plot_trace_compat_v1 import plotting_ast
from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration


def source_root():
    registry = read(ROOT / 'configs/acquisition/fm_adopted_gan_original_sources.v1.json')
    return Path(next(a['original_directory'] for a in registry['assets'] if a['id'] == 'rcgan'))


def test_entire_author_plot_preserved_except_one_legacy_missing_column_selection():
    path = source_root() / 'plotting.py'
    original = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'plot_trace')
    revised = plotting_ast(path).body[0]
    target = next(n for n in ast.walk(revised) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'da_ll')
    assert ast.dump(target.value) == ast.dump(ast.parse("da.reindex(columns=['epoch', 'll', 'real_ll']).dropna()", mode='eval').body)
    target.value = copy.deepcopy(next(n for n in ast.walk(original)
                                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                    and n.targets[0].id == 'da_ll').value)
    assert ast.dump(revised) == ast.dump(original)


def test_unknown_plot_expression_is_not_silently_patched(tmp_path):
    path = tmp_path / 'plotting.py'
    path.write_text('def plot_trace():\n    da_ll = da.fillna(0)\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unknown'):
        plotting_ast(path)


def test_recovery_preserves_every_canonical_seed_and_full_budget():
    q = read(ROOT / 'configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json')
    old = read(ROOT / q['prior_attempt_queue'])
    verify_registration(q)
    assert [(j['id'], j['seed']) for j in q['jobs']] == [(j['id'], j['seed']) for j in old['jobs']]
    assert {j['output_directory'] for j in q['jobs']}.isdisjoint(j['output_directory'] for j in old['jobs'])
    assert not q['recovery_seeds_are_additional_independent_repeats']
    for field in ['epochs', 'D_updates_per_epoch', 'G_updates_per_epoch', 'full_evaluations_per_job',
                  'sigma_maximum_updates_per_evaluation', 'full_split_shapes', 'frozen_native_packages']:
        assert q[field] == old[field]


def test_training_and_evaluation_ast_unchanged_except_plot_installation_and_receipt():
    old = ast.parse((ROOT / 'scripts/flow_matching/rcgan_native_training_runtime_v1.py').read_text(encoding='utf-8'))
    new = ast.parse((ROOT / 'scripts/flow_matching/rcgan_native_training_runtime_v2.py').read_text(encoding='utf-8'))
    run = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Import) and n.names[0].name == 'plotting')
    assert len(run.body[start:start+4]) == 4
    del run.body[start:start+4]
    call = next(n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'dict' and any(k.arg == 'plotting_compatibility_receipt' for k in n.keywords))
    call.keywords = [k for k in call.keywords if k.arg != 'plotting_compatibility_receipt']
    assert ast.dump(new) == ast.dump(old)
