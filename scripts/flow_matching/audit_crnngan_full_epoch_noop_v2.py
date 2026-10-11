"""Keep host-only process monitoring imports outside the frozen TF1 worker."""
import ast
from pathlib import Path
from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base


def execution_tree():
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    imports = [n for n in main.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if len(imports) != 6:
        raise ValueError('All six host-only controller imports must be separated')
    main.body = [n for n in main.body if n not in imports]
    worker_branch = next(n for n in main.body if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                         and n.test.attr == 'worker')
    main.body[main.body.index(worker_branch)+1:main.body.index(worker_branch)+1] = imports
    strings = [n for n in ast.walk(main) if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and ";runpy.run_module('scripts.flow_matching.audit_crnngan_full_epoch_noop_v1',run_name='__main__')" in n.value]
    if len(strings) != 1:
        raise ValueError('One unchanged complete native worker dispatcher required')
    strings[0].value = strings[0].value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v1',
                                              'scripts.flow_matching.audit_crnngan_full_epoch_noop_v2')
    return ast.fix_missing_locations(tree)


if __name__ == '__main__':
    exec(compile(execution_tree(), base.__file__, 'exec'), dict(__name__='__main__', __file__=base.__file__))
