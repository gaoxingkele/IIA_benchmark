"""Run the original module CLI body with unchanged numerical functions.

The original worker has an inline __main__ guard, not a main function. Reuse
that exact body and the v1 metadata writer; all full scientific budgets remain.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import ast
from pathlib import Path
from scripts.flow_matching import run_cfm_metadata_safe_original_job_v1 as previous


def cli_ast(original):
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    expected = ast.dump(ast.parse("__name__ == '__main__'", mode='eval').body)
    guards = [node for node in tree.body if isinstance(node, ast.If) and ast.dump(node.test) == expected]
    if len(guards) != 1:
        raise ValueError('Exactly one original inline command-line entry required')
    return ast.fix_missing_locations(ast.Module(body=guards[0].body, type_ignores=[]))


def main():
    values = previous.namespace()
    exec(compile(cli_ast(previous.original), previous.original.__file__, 'exec'), values)


if __name__ == '__main__':
    main()
