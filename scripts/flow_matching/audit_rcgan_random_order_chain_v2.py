"""Apply the entire original RNG/order audit to all84 saved seed1104 epochs.

The original500epoch training/evaluator obligation is unchanged. This audits
every persisted epoch and author random draw; it does not execute GAN training.
"""
import ast

from scripts.flow_matching import audit_rcgan_random_order_chain_v1 as original


def full84_chain_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('verify_registration', 'worker', 'main')]
    if len(tree.body) != 3:
        raise ValueError('Every complete original chain gate, worker and controller required')
    class FrozenEpochBinding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int and value == 238:
                return 84
            if isinstance(value, str):
                return value.replace('238', '84').replace(
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v1',
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v2')
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
    source = original.ROOT / 'scripts/flow_matching/audit_rcgan_random_order_chain_v1.py'
    namespace = dict(original.__dict__)
    exec(compile(full84_chain_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
