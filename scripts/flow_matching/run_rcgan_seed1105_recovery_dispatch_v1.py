"""Run both complete native gates before scheduling the full162-epoch remainder."""
import ast
from scripts.flow_matching import run_rcgan_full366_native_audit_dispatch_v1 as original
from scripts.flow_matching.rcgan_seed1105_resume_binding_v1 import MODULES

DISPATCH_MODULES = {
    'scripts.flow_matching.audit_rcgan_random_order_chain_v3': MODULES['scripts.flow_matching.audit_rcgan_random_order_chain_v1'],
    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v4': MODULES['scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1'],
    'scripts.flow_matching.run_rcgan_native_state_recovery_v3': MODULES['scripts.flow_matching.run_rcgan_native_state_recovery_v1'],
}


def dispatch_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if {n.name for n in tree.body} != {'verify_dispatch_registration', 'main'}:
        raise ValueError('Both complete native dispatch functions required')

    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int:
                node.value = {366: 338, 134: 162}.get(node.value, node.value)
            elif isinstance(node.value, str):
                node.value = node.value.replace('366', '338').replace('134', '162')
                for old, new in DISPATCH_MODULES.items():
                    node.value = node.value.replace(old, new)
            return node

        def visit_ImportFrom(self, node):
            node.module = DISPATCH_MODULES.get(node.module, node.module)
            if node.module == 'scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1':
                if len(node.names) != 1 or node.names[0].name != 'verify_registration':
                    raise ValueError('Only original registration routing may change')
                node.module = 'scripts.flow_matching.audit_rcgan_seed1105_chain_v1'
            return node

    return ast.fix_missing_locations(Binding().visit(tree))


def bind():
    path = original.ROOT / 'scripts/flow_matching/run_rcgan_full366_native_audit_dispatch_v1.py'
    namespace = dict(original.__dict__)
    exec(compile(dispatch_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
