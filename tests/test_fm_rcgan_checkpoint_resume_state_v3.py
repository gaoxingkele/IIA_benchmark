import ast
from pathlib import Path

from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2 import full84_state_ast
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v3 import full84_actual81_trace_ast


def test_actual_trace_binding_keeps_every_full_model_and_optimizer_replay_statement():
    source = Path('scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py').read_text(encoding='utf-8')
    expected = full84_state_ast(source)
    actual = full84_actual81_trace_ast(source)
    before = [n for n in ast.walk(expected) if isinstance(n, ast.Constant)]
    after = [n for n in ast.walk(actual) if isinstance(n, ast.Constant)]
    assert len(before) == len(after)
    assert any(n.value == 84 for n in after) and any(n.value == 96929 for n in after)
    assert any(n.value == 500 for n in after)
    assert sum(type(n.value) is int and n.value == 81 for n in after) == 1
    for new, old in zip(after, before):
        value = old.value
        if type(value) is int and value == 231: assert new.value == 81
        elif isinstance(value, str):
            assert new.value == value.replace('231', '81').replace('0..230', '0..80').replace(
                'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2',
                'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v3')
        else: assert new.value == value
        new.value = value
    before_keywords = [n for n in ast.walk(expected) if isinstance(n, ast.keyword)]
    after_keywords = [n for n in ast.walk(actual) if isinstance(n, ast.keyword)]
    assert len(before_keywords) == len(after_keywords)
    assert any(n.arg == 'all84_full_model_and_optimizer_checkpoints_restored_bitwise' for n in after_keywords)
    assert any(n.arg == 'all81_persisted_native_trace_losses_reproduced_bitwise_float32' for n in after_keywords)
    assert any(n.arg == 'three_missing_original_elapsed_times_preserved' for n in after_keywords)
    for new, old in zip(after_keywords, before_keywords):
        value = old.arg
        assert new.arg == (None if value is None else value.replace('all238_', 'all84_').replace(
            'all231_', 'all81_').replace('seven_missing', 'three_missing'))
        new.arg = value
    assert ast.dump(actual) == ast.dump(expected)
