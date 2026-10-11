"""Dispatch all unchanged full native jobs through inert-target normalization."""
import ast
from scripts.flow_matching import run_crnngan_native_original_queue_v2 as original
from scripts.flow_matching.run_crnngan_bounded_noop_queue_v2 import controller_tree as prior_tree


def controller_tree():
    tree = prior_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.crnngan_bounded_noop_runtime_v1':
            node.module = 'scripts.flow_matching.crnngan_executor_cache_runtime_v1'
        value = node.s if isinstance(node, ast.Str) else getattr(node, 'value', None)
        if isinstance(value, str):
            replacement = value.replace('scripts.flow_matching.run_crnngan_bounded_noop_queue_v2',
                                        'scripts.flow_matching.run_crnngan_executor_cache_queue_v1')
            if isinstance(node, ast.Str):
                node.s = replacement
            elif isinstance(node, ast.Constant):
                node.value = replacement
    return ast.fix_missing_locations(tree)


if __name__ == '__main__':
    exec(compile(controller_tree(), original.__file__, 'exec'), dict(__name__='__main__', __file__=original.__file__))
