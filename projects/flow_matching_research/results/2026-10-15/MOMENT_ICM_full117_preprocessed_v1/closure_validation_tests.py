import json
import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_full117_preprocessing_v1 as s


def case_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    (tmp_path/'config.json').write_text('{}')
    config=dict(artifact_root=str(tmp_path/'artifacts'),state_root='state',python='python310.exe',_config_path='config.json',environment={'fixture':True})
    job=dict(dataset='fixture',splits={});records={}
    for split,n in [('TRAIN',2),('TEST',1)]:
        content='@timestamps false\n@classLabel true Älg Zebra\n@data\n1,2:3,4:Älg\n'
        if split=='TRAIN':content+='5,6:7,8:Zebra\n'
        raw=tmp_path/(split+'.ts');raw.write_text(content,encoding='utf-8')
        job['splits'][split]=dict(path=raw.name,sha256=s.sha(raw),instances=n)
        base=tmp_path/'artifacts/fixture'/split;base.mkdir(parents=True)
        np.save(base/'values.npy',np.ones((n,2,256),dtype=np.float64))
        np.save(base/'input_mask.npy',np.ones((n,256),dtype=np.float64))
        np.save(base/'labels.npy',np.asarray([1,0] if n==2 else [1],dtype=np.int64))
        np.save(base/'original_instance_rows.npy',np.arange(n,dtype=np.int64))
        records[split]=dict(instances=n,channels=2,sequence_length=256,values_shape=[n,2,256],values_dtype='float64',
            source_coordinates_copied_and_checked=n*2*256,official_instance_order_and_all_channels_retained=True,
            files_sha256={p.name:s.sha(p) for p in base.iterdir()})
    proof=dict(passed=True,job=job,configuration_sha256=s.sha(tmp_path/'config.json'),environment=config['environment'],
        strict_train_only_preprocessing=False,source_AST_reused=True,independent_preprocessing_value_reconstruction=False,
        neural_model_training_or_inference_or_SVM_performed=False,raw_data_overwritten=False,
        exact_historical_ICM_preprocessing_certified=False,records=records,original_data_standardizer_fitting_performed=True)
    path=tmp_path/'artifacts/fixture/preprocessing.json';path.write_text(json.dumps(proof))
    guard=tmp_path/'state/attempts/fixture/guard_receipt.json';guard.parent.mkdir(parents=True)
    command=['python310.exe','-B','-X','utf8','-u','-m',s.producer.MODULE,'--config','config.json','--dataset','fixture']
    guard.write_text(json.dumps(dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=s.sha(path),exact_child_handle=dict(command=command))))
    return config,job,path,guard


def test_complete_stored_arrays_and_independent_unicode_class_reconciliation(tmp_path,monkeypatch):
    config,job,_,_=case_fixture(tmp_path,monkeypatch)
    report=s.close_case(config,job)
    assert report['records']['TRAIN']['coordinates']==1024
    assert report['records']['TEST']['independent_original_labels_and_rows_match']
    assert report['independent_label_and_stored_coordinate_audit']
    assert not report['independent_resampling_and_scaling_rewrite']
    assert not report['strict_train_only_preprocessing'] and not report['original_ICM_pretraining_or_performance_certified']


@pytest.mark.parametrize('mutation',['label','row','finite','stored_byte','channels'])
def test_semantic_changes_or_changed_bytes_are_rejected(tmp_path,monkeypatch,mutation):
    config,job,path,guard=case_fixture(tmp_path,monkeypatch);base=path.parent/'TRAIN';proof=s.read(path)
    name={'label':'labels.npy','row':'original_instance_rows.npy','finite':'values.npy','stored_byte':'values.npy'}.get(mutation)
    if name:
        values=np.load(base/name)
        if mutation=='label':values[0]=0
        elif mutation=='row':values[-1]=0
        elif mutation=='finite':values[0,0,0]=np.nan
        else:values[0,0,0]=42
        np.save(base/name,values)
        if mutation!='stored_byte':proof['records']['TRAIN']['files_sha256'][name]=s.sha(base/name)
    else:proof['records']['TRAIN']['channels']=1
    path.write_text(json.dumps(proof));g=s.read(guard);g['proof_sha256']=s.sha(path);guard.write_text(json.dumps(g))
    with pytest.raises(ValueError):s.close_case(config,job)
