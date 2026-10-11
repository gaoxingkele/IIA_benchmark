"""Compare native inert-fetch reuse under the exact original CPU/BLAS runtime."""
import argparse
import ast
import os
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base

NATIVE_ENVIRONMENT = dict(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')


def host_execution_tree():
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v2 import execution_tree
    tree = execution_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v2',
                                            'scripts.flow_matching.audit_crnngan_full_epoch_noop_v4')
    popen = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess' and n.func.attr == 'Popen']
    if len(popen) != 1 or any(k.arg == 'env' for k in popen[0].keywords):
        raise ValueError('One controlled original native worker launch required')
    popen[0].keywords.append(ast.keyword(arg='env', value=ast.parse("dict(os.environ, **config['native_execution_environment'])", mode='eval').body))
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    if config['native_execution_environment'] != NATIVE_ENVIRONMENT:
        raise ValueError('Original complete native training runtime environment required')
    if args.worker:
        if any(os.environ.get(k) != v for k, v in NATIVE_ENVIRONMENT.items()):
            raise ValueError('Original native CPU/BLAS environment differs')
        base.worker(config)
        return
    exec(compile(host_execution_tree(), base.__file__, 'exec'), dict(__name__='__main__', __file__=base.__file__, os=os))


if __name__ == '__main__':
    main()
