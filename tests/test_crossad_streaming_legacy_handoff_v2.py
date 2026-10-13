"""Missing historical status timestamps do not weaken actual process identity."""
import ast
import inspect

import pytest

from scripts.flow_matching import run_crossad_streaming_controller_v2 as current


@pytest.mark.parametrize('change', ['pid', 'queue', 'time'])
def test_unknown_status_identity_or_queue_is_rejected(change):
    expected = dict(pid=123, create_time=456.0)
    state = dict(pid=123, queue_sha256='original')
    if change == 'pid': state['pid'] = 124
    if change == 'queue': state['queue_sha256'] = 'other'
    if change == 'time': state['create_time'] = 457.0
    assert not current.legacy_state_matches(state, expected, 'original')


def test_absent_legacy_timestamp_requires_exact_pid_and_queue_hash():
    expected = dict(pid=123, create_time=456.0)
    assert current.legacy_state_matches(dict(pid=123, queue_sha256='original'), expected, 'original')
    assert current.legacy_state_matches(dict(pid=123, queue_sha256='original', create_time=456.0), expected, 'original')


def test_all_actual_live_suspend_recheck_and_resume_logic_unchanged():
    original = ast.parse(inspect.getsource(current.previous.handoff.safe_handoff))
    tree = current.legacy_handoff_ast()
    target = ast.parse('not _legacy_state_matches(state, expected, _legacy_queue_sha256)', mode='eval').body
    class Restore(ast.NodeTransformer):
        def visit_UnaryOp(self, node):
            if ast.dump(node) == ast.dump(target):
                return ast.parse("state.get('create_time') != expected['create_time']", mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(Restore().visit(tree)) == ast.dump(original)
