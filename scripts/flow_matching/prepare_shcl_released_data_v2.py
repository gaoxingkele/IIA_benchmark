"""Correct only the independent THRE label reference; preserve the failed v1."""
import argparse
import ast
import inspect
from types import FunctionType

from scripts.flow_matching import prepare_shcl_released_data_v1 as original


def boundary_labels(labels,start,window,mode):
    # Native THRE pairs validation features with an unrelated test-label prefix.
    # The native threshold computation ignores these labels. Keep this mismatch
    # explicit; never treat THRE's returned labels as validation ground truth.
    return labels[start:start+window] if mode in ('test','thre') else labels[:window]


def corrected_loader_audit_ast():
    tree=ast.parse(inspect.getsource(original.audit_loaders)); changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_label' for t in node.targets):
            if ast.dump(node.value)!=ast.dump(ast.parse(
                    "labels[start:start+win] if mode=='test' else labels[:win]",mode='eval').body):
                raise ValueError('Exact identified failed THRE independent-reference expression required')
            node.value=ast.parse('_boundary_labels(labels,start,win,mode)',mode='eval').body; changes+=1
    if changes!=1: raise ValueError('Only one independent boundary-label reference can change')
    return ast.fix_missing_locations(tree)


def compiled_worker():
    namespace=dict(original.__dict__,_boundary_labels=boundary_labels)
    exec(compile(corrected_loader_audit_ast(),__file__,'exec'),namespace)
    return FunctionType(original.worker.__code__,dict(original.__dict__,audit_loaders=namespace['audit_loaders']))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); config=original.native.read(original.native.ROOT/args.config)
    if not args.worker: raise ValueError('Unchanged guarded data worker required')
    compiled_worker()(config,args.config)


if __name__=='__main__':main()
