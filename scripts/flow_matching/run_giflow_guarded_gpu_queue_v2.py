"""Dispatch unchanged GiFlow jobs through the existing shared native GPU guard."""
import ast
from pathlib import Path

from scripts.flow_matching import run_grasp_protocol_queue as guard
from scripts.flow_matching import giflow_native_protocol as protocol


def bound_main():
    source = Path(guard.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changes = {
        'configs/experiments/fm_grasp_protocol_queue.v2.json': 'configs/experiments/fm_giflow_native_queue.v2.json',
        'scripts.flow_matching.grasp_protocol_runner': 'scripts.flow_matching.run_giflow_native_job',
        'Full GRASP GPU queue already owned': 'Full GiFlow GPU queue already owned'
    }
    counts = dict.fromkeys(changes, 0)
    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in changes:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(changes[node.value]), node)
            return node
    main = Binding().visit(main)
    if any(count != 1 for count in counts.values()):
        raise ValueError('Frozen GPU orchestration binding changed')
    namespace = dict(vars(guard))
    namespace.update(ROOT=protocol.ROOT, complete=protocol.complete, verify=protocol.verify,
                     sha=protocol.sha, write_json=protocol.write_json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['main']


if __name__ == '__main__':
    bound_main()()
