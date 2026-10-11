"""Keep the original strict four-epoch gate for request-scoped inert fetch reuse."""
import argparse
import ast
import os
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import NATIVE_ENVIRONMENT
from scripts.flow_matching import audit_crnngan_four_complete_epochs_v1 as original


def worker_tree():
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    imports = [n for n in ast.walk(worker) if isinstance(n, ast.ImportFrom)
               and n.module == 'scripts.flow_matching.crnngan_executor_cache_runtime_v1']
    if len(imports) != 1:
        raise ValueError('One frozen four-epoch backend binding required')
    imports[0].module = 'scripts.flow_matching.crnngan_executor_cache_runtime_v2'
    tree.body = [worker]
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if config['native_execution_environment'] != NATIVE_ENVIRONMENT:
        raise ValueError('Exact original CPU/BLAS environment required')
    if args.worker:
        if any(os.environ.get(k) != v for k, v in NATIVE_ENVIRONMENT.items()):
            raise ValueError('Frozen original native environment differs')
        namespace = dict(vars(original))
        exec(compile(worker_tree(), original.__file__, 'exec'), namespace)
        namespace['worker'](config)
        return
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import host_execution_tree
    from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base
    tree = host_execution_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v4',
                                            'scripts.flow_matching.audit_crnngan_scoped_cache_four_epochs_v2')
            if node.value == 'full_epoch_noop_equivalence.json':
                node.value = 'four_complete_epoch_executor_audit.json'
    exec(compile(tree, base.__file__, 'exec'), dict(__name__='__main__', __file__=base.__file__, os=os))


if __name__ == '__main__':
    main()
