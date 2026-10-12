import ast
import inspect

from scripts.flow_matching import prepare_shcl_released_data_v1 as original
from scripts.flow_matching import prepare_shcl_released_data_v2 as correction
from scripts.flow_matching import run_shcl_released_data_preparation_v1 as controller
from scripts.flow_matching import run_shcl_released_data_preparation_v2 as corrected_controller


def test_THRE_reference_uses_native_window_indexed_test_prefix_and_keeps_dummy_train_val_labels():
    labels=[0,0,1,1,0,1,0,0]
    assert correction.boundary_labels(labels,2,2,'thre')==[1,1]
    assert correction.boundary_labels(labels,2,2,'test')==[1,1]
    assert correction.boundary_labels(labels,2,2,'train')==[0,0]
    assert correction.boundary_labels(labels,2,2,'val')==[0,0]


def test_only_independent_expected_label_expression_changes_in_full_loader_audit():
    tree=correction.corrected_loader_audit_ast(); changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_label' for t in node.targets):
            assert ast.dump(node.value)==ast.dump(ast.parse('_boundary_labels(labels,start,win,mode)',mode='eval').body)
            node.value=ast.parse("labels[start:start+win] if mode=='test' else labels[:win]",mode='eval').body
            changed+=1
    assert changed==1 and ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(original.audit_loaders)))


def test_all_original_controller_guards_and_full_dispatch_arguments_unchanged():
    tree=corrected_controller.controller_ast(); changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.prepare_shcl_released_data_v2':
            node.value='scripts.flow_matching.prepare_shcl_released_data_v1'; changed+=1
    assert changed==1 and ast.dump(tree)==ast.dump(ast.parse(inspect.getsource(controller.main)))
