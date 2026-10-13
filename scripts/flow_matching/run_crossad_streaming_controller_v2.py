"""Retain exact live process checks for a legacy state lacking create_time.

The failed v1 transition remains preserved. Actual PID/time/full-command checks
before and after suspension remain unchanged; legacy state additionally binds
the exact PID and immutable full139 queue hash at both idle boundaries.
"""
import ast
import inspect
from types import FunctionType, SimpleNamespace

from scripts.flow_matching import run_crossad_streaming_controller_v1 as previous


def legacy_state_matches(state, expected, queue_sha256):
    return (state.get('pid') == expected['pid'] and state.get('queue_sha256') == queue_sha256
        and ('create_time' not in state or state['create_time'] == expected['create_time']))


def legacy_handoff_ast():
    tree = ast.parse(inspect.getsource(previous.handoff.safe_handoff)); changed = 0
    expected = ast.parse("state.get('create_time') != expected['create_time']", mode='eval').body
    class Legacy(ast.NodeTransformer):
        def visit_Compare(self, node):
            nonlocal changed
            if ast.dump(node) == ast.dump(expected):
                changed += 1
                return ast.parse('not _legacy_state_matches(state, expected, _legacy_queue_sha256)', mode='eval').body
            return self.generic_visit(node)
    tree = Legacy().visit(tree)
    if changed != 1:
        raise ValueError('Exactly one legacy status-file timestamp check may be adapted')
    return ast.fix_missing_locations(tree)


def main():
    args = previous.sys.argv
    config = previous.json.loads((previous.ROOT / args[args.index('--config')+1]).read_text(encoding='utf-8'))
    namespace = dict(previous.handoff.__dict__, _legacy_state_matches=legacy_state_matches,
        _legacy_queue_sha256=config['original_queue_sha256'])
    exec(compile(legacy_handoff_ast(), __file__, 'exec'), namespace)
    proxy = SimpleNamespace(**dict(previous.handoff.__dict__, safe_handoff=namespace['safe_handoff']))
    def idle_allowed(role, state, output_exists, descendants):
        return (legacy_state_matches(state, config['previous_owner'], config['original_queue_sha256'])
            and previous.idle_allowed(role, state, output_exists, descendants))
    FunctionType(previous.main.__code__, dict(previous.main.__globals__,
        handoff=proxy, idle_allowed=idle_allowed))()


if __name__ == '__main__':
    main()
