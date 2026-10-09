"""Exact native dispatch with a separately configured guarded CPU mutex."""
import ast
from pathlib import Path
from scripts.flow_matching import run_mtsbench_stat_queue as original


def bound_main():
    source=Path(original.__file__)
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    changes=0
    class Binding(ast.NodeTransformer):
        def visit_Constant(self,value):
            nonlocal changes
            if value.value=='experiments/runs/fm_heavy_recovery_cpu.lock':
                changes+=1
                return ast.copy_location(ast.Subscript(value=ast.Name(id='queue',ctx=ast.Load()),
                          slice=ast.Constant('resource_lock'),ctx=ast.Load()),value)
            return value
    node=Binding().visit(node)
    if changes!=1:
        raise ValueError('Original native mutex binding changed')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['main']


if __name__=='__main__':
    bound_main()()
