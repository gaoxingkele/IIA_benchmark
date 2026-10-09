"""Same resource guard with the reviewed cross-version diagnostic worker."""
import ast
from pathlib import Path

from scripts.flow_matching import run_giflow_cuda_memory_probe as previous


def bound_main():
    tree = ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    replacements = 0
    class WorkerBinding(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal replacements
            if node.value == 'scripts.flow_matching.giflow_cuda_memory_probe':
                replacements += 1
                return ast.copy_location(ast.Constant('scripts.flow_matching.giflow_cuda_memory_probe_v2'), node)
            return node
    function = WorkerBinding().visit(function)
    if replacements != 1:
        raise ValueError('Original memory guard worker binding changed')
    namespace = dict(vars(previous))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(previous.__file__), 'exec'), namespace)
    return namespace['main']


if __name__ == '__main__':
    bound_main()()
