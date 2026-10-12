"""Bind the complete native recovery to366 verified +134 remaining epochs."""
import ast


def cached_resource_api():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api = resource_api()
    api['resource_snapshot'] = resource_snapshot
    return api


def full_resume_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not tree.body:
        raise ValueError('Complete original full-state recovery functions required')
    class Binding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int:
                return {238: 366, 262: 134}.get(value, value)
            if isinstance(value, str):
                return value.replace('238', '366').replace('262', '134').replace('231', '358').replace(
                    'epoch_237', 'epoch_365').replace('__state_recovery_v1', '__state_recovery_v3').replace(
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v1',
                    'scripts.flow_matching.run_rcgan_native_state_recovery_v3')
            return value
        def visit_Constant(self, n):
            n.value = self.value(n.value)
            return n
        def visit_Num(self, n):
            n.n = self.value(n.n)
            return n
        def visit_Str(self, n):
            n.s = self.value(n.s)
            return n
        def visit_ImportFrom(self, n):
            if n.module == 'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1':
                if len(n.names) != 1 or n.names[0].name != 'verify_registration':
                    raise ValueError('The complete native state verifier must be bound')
                return ast.Assign(targets=[ast.Name(id=n.names[0].asname or 'verify_registration', ctx=ast.Store())],
                    value=ast.Name(id='state_verify_registration', ctx=ast.Load()))
            if n.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v1':
                n.module = 'scripts.flow_matching.rcgan_native_training_resume_runtime_v3'
            if n.module == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
                if len(n.names) != 1 or n.names[0].name != 'resource_api':
                    raise ValueError('Only the host telemetry API binding may change')
                return ast.Assign(targets=[ast.Name(id='resource_api', ctx=ast.Store())],
                    value=ast.Name(id='_cached_resource_api', ctx=ast.Load()))
            return n
    return ast.fix_missing_locations(Binding().visit(tree))
