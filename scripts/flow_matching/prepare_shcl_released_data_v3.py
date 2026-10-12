"""Full same52 native data audits without two unused Torch imports; same caps."""
import argparse
import ast
import inspect
from types import FunctionType,ModuleType

from scripts.flow_matching import prepare_shcl_released_data_v1 as original
from scripts.flow_matching.prepare_shcl_released_data_v2 import corrected_loader_audit_ast,boundary_labels


def data_module_ast(source):
    tree=ast.parse(source); kept=[]; removed=0
    for node in tree.body:
        if isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='torch':
            removed+=1
        elif isinstance(node,ast.ImportFrom) and node.module=='torch.utils.data' and len(node.names)==1 and node.names[0].name=='DataLoader':
            removed+=1
        else: kept.append(node)
    if removed!=2: raise ValueError('Exactly two unused full-data audit imports required')
    for node in kept:
        if isinstance(node,ast.ClassDef) and any(isinstance(n,ast.Name) and n.id in ('torch','DataLoader') for n in ast.walk(node)):
            raise ValueError('Original dataset classes now require Torch; review a new source version')
    tree.body=kept
    return ast.fix_missing_locations(tree)


def data_module(source):
    module=ModuleType('_iia_shcl_original_dataset_classes'); module.__file__=str(source)
    exec(compile(data_module_ast(source.read_text(encoding='utf-8')),str(source),'exec'),module.__dict__)
    return module


def loader_audit_ast():
    tree=corrected_loader_audit_ast(); body=tree.body[0].body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='spec' for t in n.targets))
    expected=ast.parse("spec=importlib.util.spec_from_file_location('_iia_shcl_native_data_loader',str(source))\n"
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)").body
    if [ast.dump(n) for n in body[start:start+3]]!=[ast.dump(n) for n in expected]:
        raise ValueError('Only the unused Torch import path of the native data module can change')
    body[start:start+3]=ast.parse('module=_data_module(source)').body
    return ast.fix_missing_locations(tree)


def compiled_worker():
    namespace=dict(original.__dict__,_boundary_labels=boundary_labels,_data_module=data_module)
    exec(compile(loader_audit_ast(),__file__,'exec'),namespace)
    return FunctionType(original.worker.__code__,dict(original.__dict__,audit_loaders=namespace['audit_loaders']))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); config=original.native.read(original.native.ROOT/args.config)
    if not args.worker: raise ValueError('Full original guarded data worker required')
    compiled_worker()(config,args.config)


if __name__=='__main__':main()
