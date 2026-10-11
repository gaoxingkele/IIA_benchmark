import ast
from pathlib import Path

import pytest

from scripts.flow_matching.rcgan_seed1104_resume_binding_v2 import full_resume_ast


@pytest.mark.parametrize('path', ['run_rcgan_native_state_recovery_v1.py', 'rcgan_native_training_resume_runtime_v1.py'])
def test_complete_original_resume_retained_with_exact_epoch_and_state_binding(path):
    source = Path('scripts/flow_matching', path).read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)]
    actual = full_resume_ast(source)
    class RemoveBindings(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1':
                return None
            if node.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v2':
                node.module = 'scripts.flow_matching.rcgan_native_training_resume_runtime_v1'
            return node
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'state_verify_registration':
                assert len(node.targets) == 1 and node.targets[0].id in ('verify_state', 'verify_registration')
                return None
            return self.generic_visit(node)
    expected = RemoveBindings().visit(expected)
    actual = RemoveBindings().visit(actual)
    before = [n for n in ast.walk(expected) if isinstance(n, ast.Constant)]
    after = [n for n in ast.walk(actual) if isinstance(n, ast.Constant)]
    assert len(before) == len(after)
    assert any(n.value == 500 for n in after)
    assert any(n.value == 84 for n in after) and any(n.value == 416 for n in after)
    for new, old in zip(after, before):
        value = old.value
        if type(value) is int: assert new.value == {238: 84, 262: 416}.get(value, value)
        elif isinstance(value, str):
            assert new.value == value.replace('238', '84').replace('262', '416').replace('231', '81').replace(
                'epoch_237', 'epoch_083').replace('__state_recovery_v1', '__state_recovery_v2').replace(
                'scripts.flow_matching.run_rcgan_native_state_recovery_v1',
                'scripts.flow_matching.run_rcgan_native_state_recovery_v2')
        else: assert new.value == value
        new.value = value
    assert ast.dump(actual) == ast.dump(expected)


def test_missing_resume_functions_are_rejected():
    with pytest.raises(ValueError): full_resume_ast('import sys')
