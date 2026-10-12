"""Restore every366 native checkpoint and reproduce all358 actual trace rows.

Eight missing elapsed times remain missing; both segments' actual clocks are
segment elapsed times, never reconstructed as a continuous original clock.
"""
import ast

from scripts.flow_matching import audit_rcgan_checkpoint_resume_state_v1 as original
from scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1 import TRACE_EPOCHS, verify_registration


def actual_trace_epochs():
    return list(TRACE_EPOCHS)


def full366_actual358_trace_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('verify_registration', 'worker', 'main')]
    if len(tree.body) != 3:
        raise ValueError('All original native checkpoint verification functions are required')
    trace_bindings = 0
    class Binding(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal trace_bindings
            if ast.dump(node) == ast.dump(ast.parse('list(range(231))', mode='eval').body):
                trace_bindings += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='actual_trace_epochs', ctx=ast.Load()), args=[], keywords=[]), node)
            return self.generic_visit(node)
        def value(self, value):
            if type(value) is int:
                return {238: 366, 231: 358}.get(value, value)
            if isinstance(value, str):
                return value.replace('epoch_237', 'epoch_365').replace('238', '366').replace('231', '358').replace(
                    '0..230', '0..80 + 84..360; eight missing rows retained').replace('seven', 'eight').replace(
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1',
                    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4')
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
            if node.arg:
                node.arg = node.arg.replace('all238_', 'all366_').replace('all231_', 'all358_').replace('seven_missing', 'eight_missing')
            return self.generic_visit(node)
    tree = Binding().visit(tree)
    if trace_bindings != 1:
        raise ValueError('Bind exactly the source trace-index assertion; never fabricate missing rows')
    return ast.fix_missing_locations(tree)


def bind():
    source = original.ROOT / 'scripts/flow_matching/audit_rcgan_checkpoint_resume_state_v1.py'
    namespace = dict(original.__dict__, verify_chain_registration=verify_registration,
                     actual_trace_epochs=actual_trace_epochs)
    exec(compile(full366_actual358_trace_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
