"""Verify every full seed1104 checkpoint and next native update after84epochs."""
import ast

from scripts.flow_matching import audit_rcgan_checkpoint_resume_state_v1 as original
from scripts.flow_matching.audit_rcgan_random_order_chain_v2 import bind as chain_binding


def full84_state_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if {node.name for node in tree.body} != {'verify_registration', 'reachable_ops',
            'assert_no_random_transition', 'compare_loss', 'worker', 'main'}:
        raise ValueError('All complete original checkpoint, loss and transition gates required')
    class FrozenEpochBinding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int and value == 238: return 84
            if isinstance(value, str):
                return value.replace('238', '84').replace('epoch_237', 'epoch_083').replace(
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1',
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2')
            return value
        def visit_Constant(self, node):
            node.value = self.value(node.value)
            return node
        def visit_Num(self, node):
            node.n = self.value(node.n)
            return node
        def visit_Str(self, node):
            node.s = self.value(node.s)
            return node
    return ast.fix_missing_locations(FrozenEpochBinding().visit(tree))


def bind():
    source = original.ROOT / 'scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py'
    namespace = dict(original.__dict__, verify_chain_registration=chain_binding()['verify_registration'])
    exec(compile(full84_state_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
