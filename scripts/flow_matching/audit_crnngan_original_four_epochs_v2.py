"""Match native MIDI names to the original output-directory basename."""
import argparse
import ast
import os
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching import audit_crnngan_original_four_epochs_v1 as original
from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import NATIVE_ENVIRONMENT


def worker_tree():
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'glob']
    if len(calls) != 1:
        raise ValueError('One original MIDI sample observation required')
    calls[0].args[0] = ast.parse("'out-%s-%d-*.mid' % (root.name, index)", mode='eval').body
    tree.body = [worker]
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if config['native_execution_environment'] != NATIVE_ENVIRONMENT:
        raise ValueError('Original CPU/BLAS environment required')
    if args.worker:
        if any(os.environ.get(k) != v for k, v in NATIVE_ENVIRONMENT.items()):
            raise ValueError('Native original CPU/BLAS environment differs')
        namespace = dict(vars(original))
        exec(compile(worker_tree(), original.__file__, 'exec'), namespace)
        namespace['worker'](config)
        return
    tree = original.host_execution_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_original_four_epochs_v1',
                                            'scripts.flow_matching.audit_crnngan_original_four_epochs_v2')
    exec(compile(tree, original.__file__, 'exec'),
         dict(__name__='__main__', __file__=original.__file__, os=os))


if __name__ == '__main__':
    main()
