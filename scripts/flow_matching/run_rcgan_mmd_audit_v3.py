"""Original guarded full evaluator controller with corrected diagnostic scope."""
import ast
from pathlib import Path
from scripts.flow_matching import run_adopted_gan_preparation_v1 as base


def main():
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "runpy.run_module('scripts.flow_matching.audit_adopted_gan_native_v1'" in node.value:
            node.value = node.value.replace('scripts.flow_matching.audit_adopted_gan_native_v1', 'scripts.flow_matching.audit_rcgan_mmd_native_v3')
            changed += 1
    if changed != 1:
        raise ValueError('Exactly one full diagnostic evaluator entry required')
    scope = dict(base.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), base.__file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
