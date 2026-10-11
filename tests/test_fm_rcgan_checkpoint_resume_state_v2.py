import ast
from pathlib import Path

import pytest

from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2 import full84_state_ast


def test_every_original_full_checkpoint_and_optimizer_transition_statement_retained():
    source = Path('scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py').read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [n for n in expected.body if isinstance(n, ast.FunctionDef)]
    actual = full84_state_ast(source)
    before = [n for n in ast.walk(expected) if isinstance(n, ast.Constant)]
    after = [n for n in ast.walk(actual) if isinstance(n, ast.Constant)]
    assert len(after) == len(before)
    assert any(n.value == 96929 for n in after) and any(n.value == 500 for n in after)
    assert any(n.value == 'native_checkpoints/epoch_083/model' for n in after)
    for new, old in zip(after, before):
        value = old.value
        if type(value) is int and value == 238: assert new.value == 84
        elif isinstance(value, str):
            assert new.value == value.replace('238', '84').replace('epoch_237', 'epoch_083').replace(
                'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1',
                'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2')
        else: assert new.value == value
        new.value = value
    assert ast.dump(actual) == ast.dump(expected)


def test_incomplete_full_state_audit_is_rejected():
    with pytest.raises(ValueError): full84_state_ast('def main(): pass')
