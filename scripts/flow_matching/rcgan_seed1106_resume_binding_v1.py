"""Bind the complete native recovery to68 saved +432 remaining original epochs.

RCGAN, Esteban et al., original author MNIST experiment. Source/data citations
stay in the frozen original five-seed model and queue. Only count/route binding
and stricter admission change; scientific statements and emergency caps stay.
"""
import ast

from scripts.flow_matching.rcgan_seed1105_resume_binding_v1 import cached_resource_api
from scripts.flow_matching.run_preserved_native_gan_retry_v1 import admission_settings

MODULES = {
    'scripts.flow_matching.audit_rcgan_random_order_chain_v1': 'scripts.flow_matching.audit_rcgan_seed1106_chain_v1',
    'scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1': 'scripts.flow_matching.audit_rcgan_seed1106_state_v1',
    'scripts.flow_matching.rcgan_native_training_resume_runtime_v1': 'scripts.flow_matching.rcgan_seed1106_resume_runtime_v1',
    'scripts.flow_matching.run_rcgan_native_state_recovery_v1': 'scripts.flow_matching.run_rcgan_seed1106_recovery_v1',
}


def stronger_admission(config, settings):
    if config.get('retry_admission_commit_headroom_bytes') != 40 * 1024 ** 3:
        raise ValueError('This actual commit incident requires frozen40GiB admission')
    return admission_settings(config, settings)


def admission_ast(tree, config_name):
    reference = ast.dump(ast.parse(config_name + "['settings']", mode='eval').body)
    changes = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'api'):
            continue
        key = node.func.slice
        if isinstance(key, ast.Index):
            key = key.value
        value = key.value if isinstance(key, ast.Constant) else key.s if isinstance(key, ast.Str) else None
        if value != 'resource_slot':
            continue
        if len(node.args) != 3 or ast.dump(node.args[1]) != reference:
            raise ValueError('Only the exact original resource admission argument may change')
        node.args[1] = ast.Call(func=ast.Name(id='_retry_admission_settings', ctx=ast.Load()),
            args=[ast.Name(id=config_name, ctx=ast.Load()), node.args[1]], keywords=[])
        changes += 1
    if changes != 1:
        raise ValueError('Exactly one stronger admission call required')
    return ast.fix_missing_locations(tree)


def recovery_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not tree.body:
        raise ValueError('Every original recovery function remains required')

    class Binding(ast.NodeTransformer):
        def value(self, value):
            if type(value) is int:
                return {238: 68, 231: 61, 262: 432}.get(value, value)
            if isinstance(value, str):
                value = value.replace('238', '68').replace('231', '61').replace('262', '432').replace('epoch_237', 'epoch_067')
                for old, new in MODULES.items():
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
            node.module = MODULES.get(node.module, node.module)
            if node.module == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
                if len(node.names) != 1 or node.names[0].name != 'resource_api':
                    raise ValueError('Only host resource telemetry may be rebound')
                return ast.Assign(targets=[ast.Name(id='resource_api', ctx=ast.Store())],
                    value=ast.Name(id='_cached_resource_api', ctx=ast.Load()))
            return node

    tree = ast.fix_missing_locations(Binding().visit(tree))
    if any(node.name == 'main' for node in tree.body):
        tree = admission_ast(tree, 'config')
    return tree


def bind_original(original, filename, **overrides):
    path = original.ROOT / ('scripts/flow_matching/' + filename)
    namespace = dict(original.__dict__, _cached_resource_api=cached_resource_api,
        _retry_admission_settings=stronger_admission, **overrides)
    exec(compile(recovery_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace
