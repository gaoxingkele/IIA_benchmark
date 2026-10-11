import ast
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching import audit_saits_full_checkpoint_replay_v1 as native
from scripts.flow_matching import validate_saits_full_result_v1 as validator
from scripts.flow_matching import verify_saits_completed_full_replay_batch_v1 as review
from scripts.flow_matching.audit_saits_physio_full_checkpoint_replay_v1 import replay_ast
from scripts.flow_matching.validate_saits_legacy_physio_full_result_v1 import legacy_dataset_sha256,validation_ast
from scripts.flow_matching.verify_saits_physio_completed_full_replay_v1 import review_ast,verify_physio_identity


def function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_full_native_replay_module_is_identical_except_legacy_validator_and_worker_dispatch():
    changed=replay_ast()
    for node in ast.walk(changed):
        if isinstance(node,ast.ImportFrom) and node.module=='scripts.flow_matching.validate_saits_legacy_physio_full_result_v1':
            node.module='scripts.flow_matching.validate_saits_full_result_v1'
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.audit_saits_physio_full_checkpoint_replay_v1':
            node.value='scripts.flow_matching.audit_saits_full_checkpoint_replay_v1'
    assert ast.dump(changed)==ast.dump(ast.parse(Path(native.__file__).read_text(encoding='utf-8')))


def test_full_optimizer_rng_checkpoint_epoch_group_and_metric_validator_is_preserved():
    changed=validation_ast().body[0]
    class Undo(ast.NodeTransformer):
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='legacy_dataset_sha256':
                return ast.Subscript(value=ast.Name(id='config',ctx=ast.Load()),slice=ast.Constant(value='dataset_sha256'),ctx=ast.Load())
            return self.generic_visit(n)
    changed=Undo().visit(changed)
    assert ast.dump(changed)==ast.dump(function(ast.parse(Path(validator.__file__).read_text(encoding='utf-8')),'validate'))


def test_all_full_array_verification_and_resource_controller_logic_is_preserved():
    changed=review_ast()
    class Undo(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            if n.module=='scripts.flow_matching.validate_saits_legacy_physio_full_result_v1':n.module='scripts.flow_matching.validate_saits_full_result_v1'
            return n
        def visit_Constant(self,n):
            backwards={'patient_statistics.npz':'group_statistics.npz',
                'scripts.flow_matching.verify_saits_physio_completed_full_replay_v1':'scripts.flow_matching.verify_saits_completed_full_replay_batch_v1',
                'audit_saits_physio_full_checkpoint_replay_v1':'audit_saits_full_checkpoint_replay_v1',
                'waiting_actual_full10_native_physio_replay_receipt':'waiting_actual_full11_native_replay_receipt'}
            if isinstance(n.value,str) and n.value in backwards:n.value=backwards[n.value]
            return n
        def visit_Call(self,n):
            if (isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='native'
                    and n.func.attr=='get' and isinstance(n.args[0],ast.Constant) and n.args[0].value in ('dataset','missing_ratio')):
                return ast.Subscript(value=ast.Name(id='native',ctx=ast.Load()),slice=n.args[0],ctx=ast.Load())
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['ids','groups']:
                n.value.elts[1].slice.value='test_groups'
            return self.generic_visit(n)
    changed=Undo().visit(changed)
    old=ast.parse(Path(review.__file__).read_text(encoding='utf-8'))
    assert ast.dump(changed)==ast.dump(old)


def identities():
    config=dict(id='saits_paper_saits_physio_seed26',method='saits',phase='author_original_protocol',seed=26,diagnostic=False)
    result=dict(config=deepcopy(config),bootstrap_group='patient',test_patients=2398,dataset_sha256='original_data',
        parameters=5322806,target_count=82605,status='completed',validation_best=dict(epoch=218))
    audit=dict(feature_names=list(range(37)),splits=dict(test=dict(shape=[2398,48,37])),
        raw_archives={'original_archive':'original_digest'},dataset_sha256='original_data')
    return config,result,audit


def test_legacy_config_stays_unchanged_and_matching_explicit_hash_is_supported():
    config,result,audit=identities();before=deepcopy(config)
    assert legacy_dataset_sha256(config,result,audit)=='original_data'
    assert config==before and 'dataset_sha256' not in config
    config['dataset_sha256']='original_data'
    assert legacy_dataset_sha256(config,result,audit)=='original_data'


@pytest.mark.parametrize('change',['method','seed','id','phase','group','features','hours','shape_features','patient_count','raw','data','explicit_hash'])
def test_legacy_binding_cannot_accept_different_data_model_or_patient_population(change):
    config,result,audit=identities()
    if change=='method':config['method']='substitute'
    elif change=='seed':config['seed']=99
    elif change=='id':config['id']='other'
    elif change=='phase':config['phase']='diagnostic'
    elif change=='group':result['bootstrap_group']='month'
    elif change=='features':audit['feature_names']=list(range(35))
    elif change=='hours':audit['splits']['test']['shape'][1]=24
    elif change=='shape_features':audit['splits']['test']['shape'][2]=35
    elif change=='patient_count':result['test_patients']=10
    elif change=='raw':audit['raw_archives']={}
    elif change=='data':audit['dataset_sha256']='changed'
    elif change=='explicit_hash':config['dataset_sha256']='changed'
    with pytest.raises(ValueError):legacy_dataset_sha256(config,result,audit)


def test_full_original_patient_identity_without_later_metadata_fields():
    config,result,_=identities()
    registered=dict(id=config['id'],config='original_path')
    record=dict(id=config['id'],config='original_path',seed=26,model_parameters=5322806,
        held_out_coordinates=82605,original_full_dataset_sha256='original_data',validation_best=dict(epoch=218))
    verify_physio_identity(record,config,result,registered)
    for key,value in [('seed',27),('model_parameters',100),('held_out_coordinates',100),
        ('original_full_dataset_sha256','changed'),('validation_best',dict(epoch=1)),('id','retry_alias')]:
        wrong=dict(record);wrong[key]=value
        with pytest.raises(ValueError):verify_physio_identity(wrong,config,result,registered)
