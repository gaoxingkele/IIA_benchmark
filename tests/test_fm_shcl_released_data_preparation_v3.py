import ast
import inspect

from scripts.flow_matching import prepare_shcl_released_data_v1 as original
from scripts.flow_matching import prepare_shcl_released_data_v2 as v2
from scripts.flow_matching import prepare_shcl_released_data_v3 as v3
from scripts.flow_matching import run_shcl_released_data_preparation_v1 as controller
from scripts.flow_matching import run_shcl_released_data_preparation_v3 as corrected_controller


def test_all_native_dataset_class_bodies_and_non_Torch_imports_unchanged():
    config=original.native.read(original.native.ROOT/'configs/models/fm_shcl_released_vae_native.v1.json')
    source=original.native.verify(config)/'data_factory/data_loader.py'
    original_tree=ast.parse(source.read_text(encoding='utf-8')); reduced=v3.data_module_ast(source.read_text(encoding='utf-8'))
    retained=[n for n in original_tree.body if not (
        isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)) and not (
        isinstance(n,ast.ImportFrom) and n.module=='torch.utils.data')]
    assert len(original_tree.body)-len(retained)==2
    assert [ast.dump(n) for n in retained]==[ast.dump(n) for n in reduced.body]
    assert sum(isinstance(n,ast.ClassDef) for n in original_tree.body)==sum(isinstance(n,ast.ClassDef) for n in reduced.body)


def test_complete_data_audit_body_unchanged_except_isolated_module_import_path():
    tree=v3.loader_audit_ast(); body=tree.body[0].body
    index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_data_module')
    body[index:index+1]=ast.parse("spec=importlib.util.spec_from_file_location('_iia_shcl_native_data_loader',str(source))\n"
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)").body
    assert ast.dump(tree)==ast.dump(v2.corrected_loader_audit_ast())


def test_same_frozen_guard_and_full_worker_arguments_after_import_only_recovery():
    tree=corrected_controller.controller_ast(); changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.prepare_shcl_released_data_v3':
            node.value='scripts.flow_matching.prepare_shcl_released_data_v1'; changed+=1
    assert changed==1 and ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(controller.main)))
    predecessor=original.native.read(original.native.ROOT/'configs/datasets/fm_shcl_released_data_views.v2.json')
    current=original.native.read(original.native.ROOT/'configs/datasets/fm_shcl_released_data_views.v3.json')
    assert current['settings']==predecessor['settings']
    assert current['datasets']==predecessor['datasets'] and current['profiles']==predecessor['profiles']
