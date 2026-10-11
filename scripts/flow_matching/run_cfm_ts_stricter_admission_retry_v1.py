"""Fresh full CFM attempts with stricter controller admission only.

The complete original scientific queue and running guards remain frozen.
Only the resource_slot settings argument receives a stronger commit minimum.
"""
import ast
from pathlib import Path

from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as original
from scripts.flow_matching.run_cfm_ts_receipt_runtime_v3 import write_with_parents
from scripts.flow_matching.run_preserved_native_gan_retry_v1 import admission_settings


def controller_ast(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(functions) != 1:
        raise ValueError('Exactly one frozen complete CFM controller main required')
    tree.body = functions
    reference = ast.dump(ast.parse("queue['resource_policy']", mode='eval').body)
    changes = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'api'
                and isinstance(node.func.slice, ast.Constant)
                and node.func.slice.value == 'resource_slot'):
            if len(node.args) != 3 or ast.dump(node.args[1]) != reference:
                raise ValueError('Frozen original CFM family admission changed')
            node.args[1] = ast.Call(func=ast.Name(id='_retry_admission_settings', ctx=ast.Load()),
                args=[ast.Name(id='runtime', ctx=ast.Load()), node.args[1]], keywords=[])
            changes += 1
    if changes != 1:
        raise ValueError('Exactly one original CFM resource admission required')
    return ast.fix_missing_locations(tree)


def bind():
    path = Path(original.__file__)
    namespace = dict(original.__dict__, write=write_with_parents,
        _retry_admission_settings=admission_settings)
    exec(compile(controller_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
