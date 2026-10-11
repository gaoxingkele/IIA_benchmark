"""Verify all84 seed1104 checkpoints against all81 actually persisted trace rows."""
import ast

from scripts.flow_matching import audit_rcgan_checkpoint_resume_state_v1 as original
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2 import full84_state_ast
from scripts.flow_matching.audit_rcgan_random_order_chain_v2 import bind as chain_binding


def full84_actual81_trace_ast(source):
    tree = full84_state_ast(source)
    class ActualTraceBinding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int and value == 231: return 81
            if isinstance(value, str):
                return value.replace('231', '81').replace('0..230', '0..80').replace(
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v2',
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v3')
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
        def visit_keyword(self, node):
            if node.arg is not None:
                node.arg = node.arg.replace('all238_', 'all84_').replace('all231_', 'all81_').replace(
                    'seven_missing', 'three_missing')
            return self.generic_visit(node)
    return ast.fix_missing_locations(ActualTraceBinding().visit(tree))


def bind():
    source = original.ROOT / 'scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py'
    namespace = dict(original.__dict__, verify_chain_registration=chain_binding()['verify_registration'])
    exec(compile(full84_actual81_trace_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
