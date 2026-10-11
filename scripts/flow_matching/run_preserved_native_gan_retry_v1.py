"""Stricter admission for fresh full-budget, verified-state GAN attempts.

Only the controller's family-lock resource admission argument is adapted.
Frozen native workers, completed-stage proofs, budgets and emergency caps keep
their original implementations. Existing attempts are never resumed in place.
"""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import sys


def admission_settings(config, original_settings):
    requested = config['retry_admission_commit_headroom_bytes']
    if (type(requested) is not int or
            requested <= original_settings['minimum_commit_headroom_bytes']):
        raise ValueError('Retry admission must strictly increase original commit headroom')
    settings = deepcopy(original_settings)
    settings['minimum_commit_headroom_bytes'] = requested
    return settings


def controller_ast(source, family):
    if family not in ('rcgan', 'timegan'):
        raise ValueError('Only the registered full-budget GAN controllers are supported')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(functions) != 1:
        raise ValueError('Exactly one frozen controller main required')
    tree.body = functions
    expected = "config['settings']" if family == 'rcgan' else "queue['settings']"
    reference = ast.dump(ast.parse(expected, mode='eval').body)
    matches = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'api'
                and isinstance(node.func.slice, ast.Constant)
                and node.func.slice.value == 'resource_slot'):
            if len(node.args) != 3 or ast.dump(node.args[1]) != reference:
                raise ValueError('Frozen family admission call changed')
            node.args[1] = ast.Call(func=ast.Name(id='_retry_admission_settings', ctx=ast.Load()),
                args=[ast.Name(id='config', ctx=ast.Load()), node.args[1]], keywords=[])
            matches += 1
    if matches != 1:
        raise ValueError('Exactly one family resource admission call required')
    return ast.fix_missing_locations(tree)


def bind(family):
    if family == 'rcgan':
        from scripts.flow_matching import run_rcgan_native_state_recovery_v1 as original
    elif family == 'timegan':
        from scripts.flow_matching import run_timegan_stage_resume_v2 as original
    else:
        raise ValueError('Unsupported GAN family')
    source = Path(original.__file__).read_text(encoding='utf-8')
    namespace = dict(original.__dict__, _retry_admission_settings=admission_settings)
    exec(compile(controller_ast(source, family), original.__file__, 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--family', required=True, choices=['rcgan', 'timegan'])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    bind(args.family)['main']()


if __name__ == '__main__':
    main()
