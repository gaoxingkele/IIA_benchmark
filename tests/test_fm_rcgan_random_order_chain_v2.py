import ast
from pathlib import Path

import pytest

from scripts.flow_matching.audit_rcgan_random_order_chain_v2 import full84_chain_ast


def test_entire_original_full_draw_protocol_is_retained_with_only_epoch_binding():
    source = Path('scripts/flow_matching/audit_rcgan_random_order_chain_v1.py').read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('verify_registration', 'worker', 'main')]
    actual = full84_chain_ast(source)
    constants = [n for n in ast.walk(actual) if isinstance(n, ast.Constant)]
    assert sum(type(n.value) is int and n.value == 84 for n in constants) == 4
    assert any(n.value == 500 for n in constants)
    assert any(n.value == 4984 for n in constants)
    original_constants = [n for n in ast.walk(expected) if isinstance(n, ast.Constant)]
    assert len(constants) == len(original_constants)
    for node, original in zip(constants, original_constants):
        value = original.value
        if type(value) is int and value == 238:
            assert node.value == 84
        elif isinstance(value, str):
            assert node.value == value.replace('238', '84').replace(
                'scripts.flow_matching.audit_rcgan_random_order_chain_v1',
                'scripts.flow_matching.audit_rcgan_random_order_chain_v2')
        else:
            assert node.value == value
        node.value = value
    assert ast.dump(actual) == ast.dump(expected)


def test_missing_original_audit_function_is_rejected():
    with pytest.raises(ValueError): full84_chain_ast('def main(): pass')
