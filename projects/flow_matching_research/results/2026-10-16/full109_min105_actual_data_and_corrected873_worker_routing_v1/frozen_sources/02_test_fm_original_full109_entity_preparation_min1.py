"""Data protocol tests; synthetic signals never count as benchmark results."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import prepare_original_full_entity_tsad_min105_v3 as s


def test_common_split_retains_every_original_group_guard_and_recipe_except_minimum():
    a,b=s.old.registration(),s.registration()
    for key in ['jobs','required_entities','total_entities','settings','resource_lock','environment','label_controls']:
        assert a[key]==b[key]
    assert len(s.verify(b))==109 and set(s.CHANGED)=={'SMAP_D-12','MSL_T-9'}
    assert {k:v for k,v in a['recipe'].items() if k not in ['minimum_validation_points','short_training_series']}=={
        k:v for k,v in b['recipe'].items() if k not in ['minimum_validation_points','short_training_series']}
    assert b['recipe']['minimum_validation_points']==105 and b['recipe']['purge_points']==100
    assert a['artifact_root']!=b['artifact_root'] and not b['previous_model_scores_reusable_as_same_protocol']


@pytest.mark.parametrize('fault',['drop_entity','purge','validation','memory','device','old_scores'])
def test_protocol_or_original_guard_mutation_rejected(fault):
    c=s.registration()
    if fault=='drop_entity':c['jobs'].pop()
    elif fault=='purge':c['recipe']['purge_points']=95
    elif fault=='validation':c['recipe']['minimum_validation_points']=100
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='device':c['python']='other_python'
    else:c['previous_model_scores_reusable_as_same_protocol']=True
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('length,entity,fit,cut',[(312,'SMAP_D-12',107,207),(439,'MSL_T-9',234,334),(682,'SMAP_A-1',445,545),(1000,'MSL_A-1',700,800)])
def test_short_entities_use_only_their_new_fit_and_keep_every_test_point(length,entity,fit,cut):
    rng=np.random.default_rng(13);train=rng.normal(size=(length,3));train[:,1]=3.33
    test=rng.normal(size=(217,3));labels=np.arange(217,dtype=np.uint8)%2
    normalize=s.old.normalizer();old_c=s.old.registration();new_c=s.registration()
    a,la,ra=normalize(train,test,labels,old_c['recipe']);b,lb,rb=normalize(train,test,labels,new_c['recipe'])
    ra['entity_id']=rb['entity_id']=entity
    assert rb['fit_interval']==[0,fit] and rb['purged_interval']==[fit,cut] and rb['validation_interval']==[cut,length]
    assert np.array_equal(b['test'],((test-np.asarray(rb['mean']))/np.where(np.asarray(rb['std'])>0,rb['std'],1)).astype(np.float32))
    assert s.old.independent_arrays(dict(train=train,test=test),rb,b)['every_normalized_coordinate_rebuilt']
    report=s.compare_entity(ra,rb,dict(a,labels=la),dict(b,labels=lb))
    assert report['split_changed']==(entity in s.CHANGED) and report['test_points']==217
    assert np.all(b['train'][:,1]==0)


def example():
    r=dict(entity_id='SMAP_A-1',fit_interval=[0,107],purged_interval=[107,207],validation_interval=[207,312],mean=[0.],std=[1.],median=[0.])
    arrays={role:np.zeros((n,1),dtype=np.float32) for role,n in [('train',107),('validation',105),('test',217)]}
    arrays['labels']=np.zeros(217,dtype=np.uint8)
    return r,arrays


@pytest.mark.parametrize('fault',['train','validation','test','labels','signed_zero','dtype','scaler','purge','identity'])
def test_every_unaffected_input_label_statistic_and_binary_float_is_checked(fault):
    r,a=example();q,b=deepcopy(r),deepcopy(a)
    if fault in ['train','validation','test','labels']:b[fault].flat[-1]=1
    elif fault=='signed_zero':b['test'].flat[-1]=-0.
    elif fault=='dtype':b['train']=b['train'].astype(np.float64)
    elif fault=='scaler':q['mean']=[1.]
    elif fault=='purge':q['purged_interval']=[106,207]
    else:q['entity_id']='other'
    with pytest.raises(ValueError):s.compare_entity(r,q,a,b)


def test_all109_actual_raw_lengths_support_common105_without_changing_other107():
    c=s.registration();differences=[]
    for job in c['jobs']:
        if job['loader']=='npy':length=job['raw']['train']['shape'][0]
        else:length=job['raw']['train']['physical_data_lines']
        old_cut=min(int(length*.8),length-100);cut=min(int(length*.8),length-105)
        assert cut-100>=105 and length-cut>=105
        if cut!=old_cut:differences.append(job['dataset']+'_'+job['entity'])
    assert set(differences)==set(s.CHANGED)


def test_cold_registration_does_not_import_numeric_frameworks():
    code='from scripts.flow_matching import prepare_original_full_entity_tsad_min105_v3 as s;import sys;s.verify(s.registration());assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,text=True,capture_output=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr


def test_completed_previous_data_proof_is_actual_and_retains_all_groups():
    binding=s.previous_binding()
    assert sum(len(r['entities']) for r in binding['records'].values())==109
    assert binding['every_numeric_coordinate_proven'] and not binding['full_NASA_source_inductivity_certified']
