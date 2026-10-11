"""Python3.7-compatible native complete35-job dispatcher binding."""
import ast
from pathlib import Path
from scripts.flow_matching import run_crnngan_native_original_queue_v2 as original


def controller_tree():
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == 'scripts.flow_matching.crnngan_native_training_runtime_v1']
    if len(imports) != 1:
        raise ValueError('One complete native worker import required')
    imports[0].module = 'scripts.flow_matching.crnngan_bounded_noop_runtime_v1'
    matches = []
    for node in ast.walk(tree):
        value = node.s if isinstance(node, ast.Str) else getattr(node, 'value', None)
        if isinstance(value, str) and ";runpy.run_module('scripts.flow_matching.run_crnngan_native_original_queue_v2',run_name='__main__')" in value:
            matches.append(node)
            replacement = value.replace('scripts.flow_matching.run_crnngan_native_original_queue_v2',
                                        'scripts.flow_matching.run_crnngan_bounded_noop_queue_v2')
            if isinstance(node, ast.Str):
                node.s = replacement
            else:
                node.value = replacement
    if len(matches) != 1:
        raise ValueError('One complete native child dispatcher required')
    return ast.fix_missing_locations(tree)


def main():
    exec(compile(controller_tree(), original.__file__, 'exec'), dict(__name__='__main__', __file__=original.__file__))


if __name__ == '__main__':
    main()
