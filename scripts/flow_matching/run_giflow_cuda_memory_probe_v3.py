"""Serial full-batch memory diagnostic under the same mutex as native GRASP/GiFlow."""
import ast
from pathlib import Path
from scripts.flow_matching import run_giflow_cuda_memory_probe as previous


def bound_main():
    tree = ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    counts = {'worker': 0, 'poll': 0}
    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.giflow_cuda_memory_probe':
                counts['worker'] += 1
                return ast.copy_location(ast.Constant('scripts.flow_matching.giflow_cuda_memory_probe_v2'), node)
            return node
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Subscript) and isinstance(node.func.slice, ast.Constant)
                    and node.func.slice.value == 'resource_slot'):
                counts['poll'] += 1
                node.keywords.append(ast.keyword(arg='poll_seconds', value=ast.Constant(.25)))
            return self.generic_visit(node)
    function = Binding().visit(function)
    if counts != {'worker': 1, 'poll': 1}:
        raise ValueError('Original memory guard changed')
    namespace = dict(vars(previous))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(previous.__file__), 'exec'), namespace)
    return namespace['main']


if __name__ == '__main__':
    bound_main()()
