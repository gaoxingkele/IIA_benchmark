"""Cold parent for full400 seed1106: original guards and checkpoints retained."""
import argparse
import ast
from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import ROOT, lightweight_namespace


def parent_namespace():
    values = lightweight_namespace()
    source = ast.parse((ROOT / 'scripts/flow_matching/run_cfm_checkpointed_original_job_v1.py').read_text(encoding='utf-8'))
    source.body = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_registration']
    if len(source.body) != 1:
        raise ValueError('Original full registration gate required')
    for node in ast.walk(source):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 1105:
            node.value = 1106
    exec(compile(ast.fix_missing_locations(source), 'seed1106_registration', 'exec'), values)
    tree = ast.parse((ROOT / 'scripts/flow_matching/run_cfm_checkpointed_full_retry_v1.py').read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Entire original guarded full400 producer required')
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.run_cfm_checkpointed_original_job_v1':
            node.value = 'scripts.flow_matching.run_cfm_checkpointed_seed1106_job_v1'
    exec(compile(ast.fix_missing_locations(tree), 'seed1106_full_producer', 'exec'), values)
    return values


if __name__ == '__main__':
    parent_namespace()['main']()
