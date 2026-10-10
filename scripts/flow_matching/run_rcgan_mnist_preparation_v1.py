"""Resource-guarded original full MNIST native loader/graph proof."""
import ast
from pathlib import Path

from scripts.flow_matching import run_adopted_gan_preparation_v1 as base


def main():
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    replaced = 0
    for node in ast.walk(target):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "runpy.run_module('scripts.flow_matching.audit_adopted_gan_native_v1'" in node.value:
                node.value = node.value.replace('scripts.flow_matching.audit_adopted_gan_native_v1',
                                                'scripts.flow_matching.audit_rcgan_mnist_native_v1')
                replaced += 1
    if replaced != 1:
        raise ValueError('Exactly one native child module binding is required')
    scope = dict(base.__dict__)
    exec(compile(ast.Module(body=[target], type_ignores=[]), base.__file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
