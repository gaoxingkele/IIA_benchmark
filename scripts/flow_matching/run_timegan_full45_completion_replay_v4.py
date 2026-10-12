"""Keep full numerical validation in the original guarded native replay child.

The parent discovers candidate metadata without importing numerical packages.
The unchanged original worker repeats complete result/array validation before
all23 checkpoint and all20 evaluator bitwise replays. No passing gate moves out
of the worker or becomes optional.
"""
import ast

from scripts.flow_matching import run_timegan_full45_completion_replay_v3 as original


def read_candidate_metadata(route):
    return original.original.base.read(original.original.base.ROOT / route['result_path'])


def main_ast():
    tree = original.main_ast()
    changes = dict(route=0, candidate=0)
    class Binding(ast.NodeTransformer):
        def value(self, value):
            if isinstance(value, str) and "runpy.run_module('scripts.flow_matching.run_timegan_full45_completion_replay_v3'" in value:
                changes['route'] += 1
                return value.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v3',
                                     'scripts.flow_matching.run_timegan_full45_completion_replay_v4')
            return value
        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node
        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node
        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(ast.parse('completed_native_result(route, queue)', mode='eval').body):
                changes['candidate'] += 1
                node.func.id = '_read_candidate_metadata'
                node.args = node.args[:1]
            return self.generic_visit(node)
    tree = Binding().visit(tree)
    if changes != dict(route=1, candidate=1):
        raise ValueError('Only the parent candidate loader and exact native worker route may change')
    return ast.fix_missing_locations(tree)


def runtime_namespace():
    namespace = original.runtime_namespace()
    namespace['_read_candidate_metadata'] = read_candidate_metadata
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    runtime_namespace()['main']()
