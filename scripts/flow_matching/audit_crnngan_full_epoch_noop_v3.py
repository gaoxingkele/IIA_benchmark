"""Run native comparison directly; host monitoring stays in the host interpreter."""
import argparse
import ast
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching import audit_crnngan_full_epoch_noop_v1 as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        base.worker(read(ROOT/args.config))
        return
    # This branch runs exclusively in the host interpreter.
    from scripts.flow_matching.audit_crnngan_full_epoch_noop_v2 import execution_tree
    tree = execution_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.audit_crnngan_full_epoch_noop_v2',
                                            'scripts.flow_matching.audit_crnngan_full_epoch_noop_v3')
    exec(compile(tree, base.__file__, 'exec'), dict(__name__='__main__', __file__=base.__file__))


if __name__ == '__main__':
    main()
