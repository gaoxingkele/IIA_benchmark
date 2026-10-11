"""Bind exact verified84epoch native state to the remaining416of500 source epochs."""
import ast


def full_resume_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not tree.body:
        raise ValueError('Complete original full-state resume functions required')
    class EpochBinding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int:
                return {238: 84, 262: 416}.get(value, value)
            if isinstance(value, str):
                return value.replace('238', '84').replace('262', '416').replace('231', '81').replace(
                    'epoch_237', 'epoch_083').replace('__state_recovery_v1', '__state_recovery_v2').replace(
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v1',
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v2')
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
            if node.module == 'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1':
                if len(node.names) != 1 or node.names[0].name != 'verify_registration':
                    raise ValueError('Exact full checkpoint verifier binding required')
                name = node.names[0].asname or node.names[0].name
                return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Name(id='state_verify_registration', ctx=ast.Load()))
            if node.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v1':
                node.module = 'scripts.flow_matching.rcgan_native_training_resume_runtime_v2'
            return node
    return ast.fix_missing_locations(EpochBinding().visit(tree))
