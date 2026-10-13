"""Run both complete native gates before the full432 original-epoch remainder."""
import ast

from scripts.flow_matching import run_rcgan_full366_native_audit_dispatch_v1 as original
from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import MODULES, admission_ast, stronger_admission

DISPATCH_MODULES = {
    'scripts.flow_matching.audit_rcgan_random_order_chain_v3': MODULES['scripts.flow_matching.audit_rcgan_random_order_chain_v1'],
    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4': MODULES['scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1'],
    'scripts.flow_matching.run_rcgan_native_state_recovery_v3': MODULES['scripts.flow_matching.run_rcgan_native_state_recovery_v1'],
}


def dispatch_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if {node.name for node in tree.body} != {'verify_dispatch_registration', 'main'}:
        raise ValueError('Both complete native dispatch functions required')

    class Binding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int:
                return {366: 68, 134: 432}.get(value, value)
            if isinstance(value, str):
                value = value.replace('366', '68').replace('134', '432')
                for old, new in DISPATCH_MODULES.items():
                    value = value.replace(old, new)
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

        def visit_ImportFrom(self, node):
            node.module = DISPATCH_MODULES.get(node.module, node.module)
            if node.module == 'scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1':
                if len(node.names) != 1 or node.names[0].name != 'verify_registration':
                    raise ValueError('Only the exact failed-source registration route changes')
                node.module = 'scripts.flow_matching.audit_rcgan_seed1106_chain_v1'
            return node

    return admission_ast(ast.fix_missing_locations(Binding().visit(tree)), 'stage_config')


def bind():
    path = original.ROOT / 'scripts/flow_matching/run_rcgan_full366_native_audit_dispatch_v1.py'
    namespace = dict(original.__dict__, _retry_admission_settings=stronger_admission)
    exec(compile(dispatch_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
