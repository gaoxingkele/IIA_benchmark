import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching import audit_crnngan_original_four_epochs_v1 as audit


def test_real_array_comparison_reports_all_differences_without_certifying_equivalence():
    old = {'a': np.array([1, 2], dtype=np.float32), 'b': np.array([3], dtype=np.float32)}
    new = {'a': np.array([1, 2], dtype=np.float32), 'b': np.array([3.5], dtype=np.float32)}
    assert audit.compare_arrays(old.__getitem__, new.__getitem__, ['a', 'b'], np) == [
        dict(name='b', max_absolute_difference=0.5)]
    assert audit.compare_arrays(old.__getitem__, old.__getitem__, ['a', 'b'], np) == []
    new['b'][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        audit.compare_arrays(old.__getitem__, new.__getitem__, ['a', 'b'], np)


def test_worker_uses_original_executor_and_stops_only_after_four_full_epochs():
    tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    original_run = [n for n in calls if isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'original'
                    and n.func.attr == 'run']
    assert len(original_run) == 1
    assert ast.dump(original_run[0].args[0]) == ast.dump(ast.Name(id='queue', ctx=ast.Load()))
    assert 'executor_cache' not in ast.unparse(worker)
    stops = [n for n in ast.walk(worker) if isinstance(n, ast.Raise)
             and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name)
             and n.exc.func.id == 'FourOriginalEpochsStop']
    assert len(stops) == 1
    enclosing = next(n for n in ast.walk(worker) if isinstance(n, ast.If) and stops[0] in n.body)
    assert ast.unparse(enclosing.test) == 'len(reports) == 4'


def test_host_preserves_original_resource_guard_and_native_environment():
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import host_execution_tree
    original = host_execution_tree()
    actual = audit.host_execution_tree()
    for node in ast.walk(actual):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_original_four_epochs_v1',
                                            'scripts.flow_matching.audit_crnngan_full_epoch_noop_v4')
            if node.value == 'original_four_epoch_repeatability.json':
                node.value = 'full_epoch_noop_equivalence.json'
            if node.value == 'running_original_four_complete_epochs':
                node.value = 'running_complete_native_epoch_equivalence'
    assert ast.dump(actual) == ast.dump(original)
