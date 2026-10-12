import ast
import inspect
import pickle
import subprocess
import sys
import zipfile

import pytest

from scripts.flow_matching import audit_full500_image_checkpoint_CPU_v2 as original
from scripts.flow_matching import audit_full813_image_checkpoint_CPU_v1 as current
from scripts.flow_matching import run_full500_image_checkpoint_CPU_audit_v2 as old_controller
from scripts.flow_matching import run_full813_image_checkpoint_CPU_audit_v1 as controller
from scripts.flow_matching.register_full813_checkpoint_CPU_audit_v1 import counts


def reverse(tree):
    class Restore(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='verify_archive_binding':
                return None
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value==813: node.value=500
            elif node.value==1187: node.value=1500
            elif isinstance(node.value,str):
                node.value=node.value.replace('full813','full500').replace('CPU_v1','CPU_v2').replace('All813','All500')
                if node.value=='resume.pt':node.value='epoch_0500.pt'
                if node.value=='source_checkpoint_receipt':node.value='checkpoint_receipt'
            return node
    return Restore().visit(tree)


def test_every_original_full_model_optimizer_environment_RNG_epoch_and_full25_gate_retained():
    for name in ('verify','worker'):
        assert ast.dump(reverse(ast.parse(inspect.getsource(getattr(current,name))))) == ast.dump(ast.parse(inspect.getsource(getattr(original,name))))


def test_cold_controller_retains_original_resource_slot_child_limits_and_terminal_checks():
    assert ast.dump(reverse(ast.parse(inspect.getsource(controller.main)))) == ast.dump(ast.parse(inspect.getsource(old_controller.main)))
    code='import scripts.flow_matching.run_full813_image_checkpoint_CPU_audit_v1,sys;assert not {"torch","numpy"}&set(sys.modules)'
    result=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')


def test_metadata_counters_read_without_executing_checkpoint_pickle(tmp_path):
    path=tmp_path/'checkpoint.pt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/data.pkl',pickle.dumps(dict(completed_epochs=813,global_steps=92682),protocol=2))
    assert counts(path)==dict(completed_epochs=813,global_steps=92682)


def test_missing_native_metadata_counters_rejected(tmp_path):
    path=tmp_path/'bad.pt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/data.pkl',pickle.dumps(dict(completed_epochs=813),protocol=2))
    with pytest.raises(ValueError):counts(path)


@pytest.mark.parametrize('change',['archive','budget','epochs','credited','same_path'])
def test_archive_binding_rejects_wrong_state_budget_or_partial_credit(monkeypatch,change):
    source=dict(path='original/resume.pt',sha256='abc')
    saved=dict(path='saved/epoch_0813_full_state.pt',sha256='abc')
    archive=dict(source_checkpoint=source,immutable_archive=saved,expected_completed_original_epochs=813,
        required_original_epochs=2000,new_optimizer_updates=0)
    config=dict(source_checkpoint_receipt=dict(source),checkpoint_receipt=dict(saved),immutable_archive_configuration='archive.json')
    if change=='archive':config['checkpoint_receipt']['sha256']='other'
    elif change=='budget':archive['required_original_epochs']=1000
    elif change=='epochs':archive['expected_completed_original_epochs']=812
    elif change=='credited':archive['new_optimizer_updates']=1
    else:config['checkpoint_receipt']=dict(source);archive['immutable_archive']=dict(source)
    monkeypatch.setattr(current,'read',lambda path:archive)
    monkeypatch.setattr(current,'verify_receipts',lambda receipts:None)
    with pytest.raises(ValueError):current.verify_archive_binding(config)
