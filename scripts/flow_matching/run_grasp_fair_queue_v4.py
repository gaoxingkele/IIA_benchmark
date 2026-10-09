"""Preserve full native jobs and guards, yielding the GPU slot between jobs."""
import ast
from pathlib import Path
from scripts.flow_matching import run_grasp_protocol_queue as original


def bound_main():
    source = Path(original.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changes = {'configs/experiments/fm_grasp_protocol_queue.v2.json': 'configs/experiments/fm_grasp_protocol_queue.v4.json',
               'scripts.flow_matching.grasp_protocol_runner': 'scripts.flow_matching.grasp_protocol_cuda_runtime'}
    counts = dict.fromkeys(changes, 0)
    yields = 0
    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in changes:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(changes[node.value]), node)
            return node
        def visit_With(self, node):
            nonlocal yields
            node = self.generic_visit(node)
            if (len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call)
                    and isinstance(node.items[0].context_expr.func, ast.Subscript)
                    and isinstance(node.items[0].context_expr.func.slice, ast.Constant)
                    and node.items[0].context_expr.func.slice.value == 'resource_slot'):
                yields += 1
                # Outside the slot: release both ownership and model first.
                return [node, ast.parse('time.sleep(queue["controller_yield_seconds"])').body[0]]
            return node
    function = Binding().visit(function)
    if any(count != 1 for count in counts.values()) or yields != 1:
        raise ValueError('Original guarded orchestration changed')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['main']


if __name__ == '__main__':
    bound_main()()
