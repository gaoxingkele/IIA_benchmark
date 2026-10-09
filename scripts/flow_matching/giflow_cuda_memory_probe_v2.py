"""Cross-version AST validation for the isolated full-batch CUDA memory diagnostic."""
import ast
from pathlib import Path

from scripts.flow_matching import giflow_cuda_memory_probe as previous


def bound_main():
    tree = ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bound_main')
    replacements = 0
    class StructuralLambda(ast.NodeTransformer):
        def visit_Compare(self, node):
            nonlocal replacements
            if (ast.unparse(node.left) == 'ast.unparse(node.value)'
                    and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == 'lambda: False'):
                replacements += 1
                return ast.copy_location(ast.parse(
                    "not (isinstance(node.value, ast.Lambda) and isinstance(node.value.body, ast.Constant) "
                    "and node.value.body.value is False and not node.value.args.args "
                    "and not node.value.args.posonlyargs and not node.value.args.kwonlyargs "
                    "and node.value.args.vararg is None and node.value.args.kwarg is None)", mode='eval').body, node)
            return self.generic_visit(node)
    function = StructuralLambda().visit(function)
    if replacements != 1:
        raise ValueError('Previous AST diagnostic guard needs review')
    namespace = dict(vars(previous))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(previous.__file__), 'exec'), namespace)
    return namespace['bound_main']()


def main():
    tree = ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    namespace = dict(vars(previous), bound_main=bound_main)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(previous.__file__), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
