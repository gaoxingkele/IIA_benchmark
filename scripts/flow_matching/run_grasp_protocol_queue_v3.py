"""Bind preserved queue dispatch to reviewed CUDA startup and recovery outputs."""
import ast
from pathlib import Path
from scripts.flow_matching import run_grasp_protocol_queue as original


def bound_main():
    source = Path(original.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    changes = {'scripts.flow_matching.grasp_protocol_runner':'scripts.flow_matching.grasp_protocol_cuda_runtime',
               'configs/experiments/fm_grasp_protocol_queue.v2.json':'configs/experiments/fm_grasp_protocol_queue.v3.json'}
    counts = {key:0 for key in changes}
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in changes:
                counts[node.value]+=1
                return ast.copy_location(ast.Constant(changes[node.value]),node)
            return node
    main = Binding().visit(main)
    if any(count!=1 for count in counts.values()):
        raise ValueError('Original dispatch changed; exact runtime binding must be reviewed')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['main']


if __name__ == '__main__':
    bound_main()()
