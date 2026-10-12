import ast
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.flow_matching.run_cfm_checkpointed_seed1106_parent_v1 import parent_namespace


def test_fresh_parent_has_no_scientific_imports_and_preserves_full_budget():
    code = """from scripts.flow_matching.run_cfm_checkpointed_seed1106_parent_v1 import parent_namespace
v=parent_namespace()
import sys
assert not any(n in sys.modules for n in ('numpy','scipy','torch'))
assert 400 in v['main'].__code__.co_consts
"""
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], check=True)


def test_parent_changes_only_seed_identity_and_worker_entrypoint():
    values = parent_namespace()
    for name, path in [('verify_registration', 'run_cfm_checkpointed_original_job_v1.py'),
                       ('main', 'run_cfm_checkpointed_full_retry_v1.py')]:
        tree = ast.parse(Path('scripts/flow_matching', path).read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        expected = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), expected)
        a, b = values[name].__code__, expected[name].__code__
        assert a.co_code == b.co_code and a.co_names == b.co_names and a.co_varnames == b.co_varnames
        # Every nested update/receipt branch is unchanged, only two bindings differ.
        from types import CodeType
        def compare(left, right):
            assert left.co_code == right.co_code and left.co_names == right.co_names
            assert len(left.co_consts) == len(right.co_consts)
            for x, y in zip(left.co_consts, right.co_consts):
                if isinstance(x, CodeType): compare(x, y)
                elif y == 1105 and type(y) is int: assert x == 1106
                elif y == 'scripts.flow_matching.run_cfm_checkpointed_original_job_v1':
                    assert x == 'scripts.flow_matching.run_cfm_checkpointed_seed1106_job_v1'
                else: assert x == y
        compare(a, b)


def test_registration_rejects_shortened_budget_or_changed_seed(tmp_path):
    values = parent_namespace()
    function = values['verify_registration']
    job = dict(id='retry', recovery_original_job_id='canonical', seed=1106, method='node',
        epochs=400, expected_optimizer_updates=400, test_points_per_trajectory=500, batch_size=258)
    canonical = dict(job, id='canonical');canonical.pop('recovery_original_job_id')
    queue = dict(jobs=[job] + [dict(id=str(i)) for i in range(284)], resource_policy={})
    config = dict(source_receipts=[],queue='x',job_id='retry',canonical_job=canonical,
        checkpoint_interval=10,resource_policy={},original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        resume_full_epoch_checkpoint=None,resume_full_epoch_checkpoint_sha256=None)
    old_read, old_verify = values['original'].read, values['original'].verify
    values['original'].read = lambda _: queue
    values['original'].verify = lambda _: None
    try:
        function(config)
        job['epochs'] = 177
        with pytest.raises(ValueError): function(config)
        job['epochs'] = 400;job['seed'] = 1105
        with pytest.raises(ValueError): function(config)
    finally:
        values['original'].read, values['original'].verify = old_read, old_verify


def test_entire_native_worker_ast_changes_only_original_seed():
    module = ast.parse(Path('scripts/flow_matching/run_cfm_checkpointed_seed1106_job_v1.py').read_text(encoding='utf-8'))
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'seed1106_ast']
    namespace = dict(ast=ast)
    exec(compile(ast.fix_missing_locations(module), 'seed1106_binding', 'exec'), namespace)
    source = Path('scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py').read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)]
    actual = namespace['seed1106_ast'](source)
    for node in ast.walk(actual):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 1106:
            node.value = 1105
    assert ast.dump(actual) == ast.dump(expected)
    with pytest.raises(ValueError): namespace['seed1106_ast']('def run(): pass')
