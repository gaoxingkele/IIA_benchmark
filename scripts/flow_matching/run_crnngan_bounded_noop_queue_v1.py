"""Bind the existing complete35-job controller to graph-local inert fetch reuse."""
import ast
from pathlib import Path

from scripts.flow_matching import run_crnngan_native_original_queue_v2 as original


def controller_tree():
    path = Path(original.__file__)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'scripts.flow_matching.crnngan_native_training_runtime_v1']
    if len(imports) != 1:
        raise ValueError('One complete native worker import required')
    imports[0].module = 'scripts.flow_matching.crnngan_bounded_noop_runtime_v1'
    strings = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and ";runpy.run_module('scripts.flow_matching.run_crnngan_native_original_queue_v2',run_name='__main__')" in node.value]
    if len(strings) != 1:
        raise ValueError('One complete native child dispatcher required')
    strings[0].value = strings[0].value.replace('scripts.flow_matching.run_crnngan_native_original_queue_v2',
                                             'scripts.flow_matching.run_crnngan_bounded_noop_queue_v1')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(__name__='__main__', __file__=original.__file__)
    exec(compile(controller_tree(), original.__file__, 'exec'), namespace)


if __name__ == '__main__':
    main()
