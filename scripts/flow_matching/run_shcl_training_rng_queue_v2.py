"""Fix two unstarted host write bindings; preserve the frozen complete audit."""
import ast
import inspect

from scripts.flow_matching import run_shcl_training_rng_queue_v1 as original
from scripts.flow_matching.timegan_native_common_v1 import write


def compiled_main():
    tree = ast.parse(inspect.getsource(original.main))
    class BindStatusWriter(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'write'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'native'):
                node.func = ast.copy_location(ast.Name(id='write', ctx=ast.Load()), node.func)
                self.count += 1
            return node
    change = BindStatusWriter()
    tree = change.visit(tree)
    if change.count != 2:
        raise ValueError('Only the two unstarted host status/receipt write bindings change')
    namespace = dict(original.__dict__, write=write)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    return namespace['main']


def main():
    compiled_main()()


if __name__ == '__main__':
    main()
