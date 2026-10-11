"""One actual TCCM capacity case retaining the original native GPU guards."""
import ast
from pathlib import Path

from scripts.flow_matching import run_full_image_cuda_capacity_queue_v2 as original


def controller_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Exactly one original native capacity controller required')
    counts = dict(cases=0, worker=0, summary=0, state=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and [getattr(n, 'value', None) for n in node.elts] == ['WT', 'FM']:
            node.elts = [ast.Constant('TCCM')]; counts['cases'] += 1
        if isinstance(node, ast.Constant):
            if node.value == 'scripts.flow_matching.probe_full_image_cuda_execution_recovery_v2':
                node.value = 'scripts.flow_matching.probe_tccm_full_image_execution_recovery_v1'; counts['worker'] += 1
            if node.value == 'both_full_native_batch32_cuda_capacity_cases_passed':
                node.value = 'full_native_tccm_batch32_capacity_passed'; counts['state'] += 1
        if isinstance(node, ast.keyword) and node.arg == 'full_native_capacity_cases':
            if not isinstance(node.value, ast.Constant) or node.value.value != 2:
                raise ValueError('Original capacity summary changed')
            node.value = ast.Constant(1); counts['summary'] += 1
    if any(n != 1 for n in counts.values()):
        raise ValueError('Exact capacity adapter structure changed: ' + str(counts))
    return ast.fix_missing_locations(tree)


def bind():
    path = Path(original.__file__)
    namespace = dict(original.__dict__)
    exec(compile(controller_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
