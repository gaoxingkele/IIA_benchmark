"""Retain the full238+262 native runtime, accepting the exact registered retry ID."""
import ast
from pathlib import Path

from scripts.flow_matching import rcgan_native_training_resume_runtime_v1 as original


def native_retry_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run']
    if len(tree.body) != 1:
        raise ValueError('Exactly the original complete native resume function required')
    matches = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and len(node.comparators) == 1
                and ast.dump(node.left) == ast.dump(ast.parse("job['id']", mode='eval').body)
                and isinstance(node.ops[0], ast.NotEq)
                and ast.dump(node.comparators[0]) == ast.dump(ast.parse("canonical['id']+'__state_recovery_v1'", mode='eval').body)):
            node.comparators[0] = ast.parse("canonical['id']+'__state_recovery_v2'", mode='eval').body
            matches += 1
    if matches != 1:
        raise ValueError('Exactly one registered retry identity comparison may change')
    return ast.fix_missing_locations(tree)


def run(config, queue, binding):
    if (config['canonical_job']['id'] != 'rcgan_mnistfull__seed1103'
            or config['recovery_job']['id'] != 'rcgan_mnistfull__seed1103__state_recovery_v2'
            or config['canonical_job']['seed'] != 1103 or config['recovery_job']['seed'] != 1103
            or config['full_original_epochs'] != 500 or config['verified_completed_epochs'] != 238
            or config['remaining_full_epochs'] != 262):
        raise ValueError('Only the exact registered seed1103 full238+262 retry is supported')
    path = Path(original.__file__)
    namespace = dict(original.__dict__)
    exec(compile(native_retry_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace['run'](config, queue, binding)


verify_result = original.verify_result
