"""Same full400 checkpointed NODE worker, bound to original seed1106."""
import ast
from scripts.flow_matching import run_cfm_checkpointed_original_job_v1 as original


def seed1106_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if {n.name for n in tree.body} != {'verify_registration', 'worker_ast', 'run'}:
        raise ValueError('All original full checkpointed worker functions required')
    class SeedBinding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 1105:
                node.value = 1106
            return node
    return ast.fix_missing_locations(SeedBinding().visit(tree))


def bind():
    path = original.original.ROOT / 'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py'
    namespace = dict(original.__dict__)
    exec(compile(seed1106_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    parser = original.argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    bind()['run'](original.original.read(original.original.ROOT / args.config))
