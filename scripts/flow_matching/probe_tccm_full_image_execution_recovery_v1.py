"""Full native batch32 TCCM capacity, never a training or performance result."""
import ast
from pathlib import Path

from scripts.flow_matching import probe_full_image_cuda_execution_recovery_v2 as original
from scripts.flow_matching.tccm_full_image_training_contract_v1 import validate_tccm_job


def capacity_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Exactly one full native CUDA capacity main required')
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'iia_benchmark.models.fm_supplemental':
            if [(n.name, n.asname) for n in node.names] != [('image_cfm_objective', None)]:
                raise ValueError('Original objective import changed')
            node.module = 'iia_benchmark.models.fm_tccm_image'
            node.names = [ast.alias(name='tccm_uniform_objective', asname='image_cfm_objective')]
            changes += 1
    if changes != 1:
        raise ValueError('Exactly one objective binding required')
    return ast.fix_missing_locations(tree)


def bind():
    source = Path(original.__file__)
    namespace = dict(original.__dict__, validate_job=validate_tccm_job)
    exec(compile(capacity_ast(source.read_text(encoding='utf-8')), str(source), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
