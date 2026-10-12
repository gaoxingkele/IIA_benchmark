"""Audit all366 native seed1104 RNG/order states across both saved segments."""
import ast

from scripts.flow_matching import audit_rcgan_random_order_chain_v1 as original
from scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1 import verify_registration


def full366_chain_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('worker', 'main')]
    if {n.name for n in tree.body} != {'worker', 'main'}:
        raise ValueError('The entire native random/order worker and guarded controller are required')
    class Binding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int and value == 238:
                return 366
            if isinstance(value, str):
                return value.replace('238', '366').replace(
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v1',
                    'scripts.flow_matching.audit_rcgan_random_order_chain_v3')
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
    return ast.fix_missing_locations(Binding().visit(tree))


def bind():
    source = original.ROOT / 'scripts/flow_matching/audit_rcgan_random_order_chain_v1.py'
    namespace = dict(original.__dict__, verify_registration=verify_registration)
    exec(compile(full366_chain_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
