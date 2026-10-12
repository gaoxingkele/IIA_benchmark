"""Bind existing complete recovery gates to seed1105's 338 saved epochs.

Only epoch counts, persisted trace length and entry-point routing change. Every
scientific statement, loss tolerance, evaluator budget and resource gate stays.
"""
import ast

MODULES = {
    'scripts.flow_matching.audit_rcgan_random_order_chain_v1': 'scripts.flow_matching.audit_rcgan_seed1105_chain_v1',
    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1': 'scripts.flow_matching.audit_rcgan_seed1105_state_v1',
    'scripts.flow_matching.rcgan_native_training_resume_runtime_v1': 'scripts.flow_matching.rcgan_seed1105_resume_runtime_v1',
    'scripts.flow_matching.run_rcgan_native_state_recovery_v1': 'scripts.flow_matching.run_rcgan_seed1105_recovery_v1',
}


def cached_resource_api():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api = resource_api()
    api['resource_snapshot'] = resource_snapshot
    return api


def recovery_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not tree.body:
        raise ValueError('All original recovery functions are required')

    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int:
                node.value = {238: 338, 231: 331, 262: 162}.get(node.value, node.value)
            elif isinstance(node.value, str):
                node.value = node.value.replace('238', '338').replace('231', '331').replace('262', '162').replace('epoch_237', 'epoch_337')
                for old, new in MODULES.items():
                    node.value = node.value.replace(old, new)
            return node

        def visit_Num(self, node):
            node.n = {238: 338, 231: 331, 262: 162}.get(node.n, node.n)
            return node

        def visit_Str(self, node):
            node.s = node.s.replace('238', '338').replace('231', '331').replace('262', '162').replace('epoch_237', 'epoch_337')
            for old, new in MODULES.items():
                node.s = node.s.replace(old, new)
            return node

        def visit_ImportFrom(self, node):
            if node.module in MODULES:
                node.module = MODULES[node.module]
            if node.module == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
                if len(node.names) != 1 or node.names[0].name != 'resource_api':
                    raise ValueError('Only host resource telemetry may be bound')
                return ast.Assign(targets=[ast.Name(id='resource_api', ctx=ast.Store())],
                                  value=ast.Name(id='_cached_resource_api', ctx=ast.Load()))
            return node

    return ast.fix_missing_locations(Binding().visit(tree))


def bind_original(original, filename, **overrides):
    path = original.ROOT / ('scripts/flow_matching/' + filename)
    namespace = dict(original.__dict__, _cached_resource_api=cached_resource_api, **overrides)
    exec(compile(recovery_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace
