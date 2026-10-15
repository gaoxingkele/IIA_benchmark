import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as s


def test_registration_covers_all109_distinct_entities_and_all_label_controls():
    c=s.registration();jobs=s.verify(c)
    assert len(jobs)==109 and len({(j['dataset'],j['entity']) for j in jobs})==109
    assert {d:sum(j['dataset']==d for j in jobs) for d in c['required_entities']}=={'SMAP':54,'MSL':27,'SMD':28}
    p2=next(j for j in jobs if j['dataset']=='SMAP' and j['entity']=='P-2')
    assert len(p2['metadata_rows'])==2
    assert sum(len(v) for v in c['label_controls'].values())==7
    assert c['raw_NASA_TEST_minmax_prescaling_disclosed'] and not c['globally_inductive_source_data_certified']
    original=s.read(s.ROOT/c['original_data_resource_policy'])
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']


@pytest.mark.parametrize('change',['jobs','settings','recipe','label_controls'])
def test_dropped_entity_reduced_resource_or_selected_label_policy_rejected(change):
    c=s.registration()
    if change=='jobs':c[change].pop()
    elif change=='settings':c[change]['minimum_commit_headroom_bytes']=1
    elif change=='recipe':c[change]['purge_points']=0
    else:c[change]['SMAP'].pop()
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('length,cut,adjusted',[(312,212,True),(439,339,True),(682,545,False),(1000,800,False)])
def test_full_short_source_series_preserved_with_frozen_100point_validation(length,cut,adjusted):
    cfg=s.registration();rng=np.random.default_rng(13)
    train=rng.normal(size=(length,3));train[:,1]=3.33
    test=rng.normal(size=(217,3));labels=np.zeros(217,dtype=np.uint8)
    arrays,targets,record=s.normalizer()(train,test,labels,cfg['recipe'])
    assert record['fit_interval']==[0,cut-100] and record['purged_interval']==[cut-100,cut]
    assert record['validation_interval']==[cut,length] and (cut!=int(length*.8))==adjusted
    assert len(arrays['train'])>=100 and len(arrays['validation'])>=100
    assert np.all(arrays['train'][:,1]==0) and np.array_equal(targets,labels)
    proof=s.independent_arrays(dict(train=train,test=test),record,arrays)
    assert proof['every_normalized_coordinate_rebuilt']
    assert proof['input_coordinates']==(length-100+217)*3


def test_normalizer_preserves_constant_source_AST_except_declared_split_rule():
    from scripts.flow_matching import prepare_tsad_constant_fit_v2 as source
    original=source.constant_scaling_ast(Path(source.original.__file__).read_text(encoding='utf-8'))
    actual=s.normalizer()
    tree=ast.parse(Path(s.__file__).read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalizer')
    # The only numerical AST mutation is assignment of cut;all normalization
    # work,repair and variance operations compile from the pinned old source.
    updates=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='node.value']
    assert len(updates)==1 and 'minimum_validation_points' in ast.unparse(updates[0])
    assert actual.__code__.co_name=='split_and_normalize' and len(original.body)==1


def label_rows():
    return [dict(num_values='8',anomaly_sequences='[[1, 3]]'),dict(num_values='8',anomaly_sequences='[[4, 6]]')]


@pytest.mark.parametrize('policy,expected',[
    ('source_first_exclusive',[1,2]),('source_first_inclusive',[1,2,3]),
    ('source_last_exclusive',[4,5]),('source_last_inclusive',[4,5,6])])
def test_conflicting_P2_rows_and_endpoint_controls_stay_separate(policy,expected):
    rows=label_rows();before=deepcopy(rows);value=s.label_array(rows,8,policy)
    assert np.flatnonzero(value).tolist()==expected and value.dtype==np.uint8
    assert rows==before


@pytest.mark.parametrize('fault',['policy','length','bounds'])
def test_invalid_label_policy_length_and_out_of_range_annotation_rejected(fault):
    rows=label_rows();length=8;policy='source_first_exclusive'
    if fault=='policy':policy='test_selected_best'
    elif fault=='length':length=9
    else:rows[0]['anomaly_sequences']='[[1, 8]]'
    with pytest.raises(ValueError):s.label_array(rows,length,policy)


def test_all_coordinate_mapping_detects_CSV_training_taken_from_original_TEST(tmp_path):
    original=np.column_stack([np.arange(10,dtype=float),np.arange(10,dtype=float)**2])
    labels=(np.arange(10)%2).astype(np.uint8);path=tmp_path/'csv_train.csv'
    pd.DataFrame(dict(timestamp=np.arange(3,8),feature_1=original[2:7,0],feature_2=original[2:7,1],is_anomaly=labels[2:7])).to_csv(path,index=False)
    inventory=dict(registered_CSV_files=[dict(dataset='SMD',entity='fixture',path=str(path),sha256=s.sha(path),role='train')])
    rows=s.compare_registered_CSV(dict(dataset='SMD',entity='fixture',loader='text'),dict(train=np.ones_like(original)*-100,test=original,labels=labels),inventory)
    assert len(rows)==1 and rows[0]['all_original_CSV_coordinates_compared']
    exact=rows[0]['exact_matching_candidates'];assert len(exact)==1
    assert exact[0]['raw_role']=='test' and exact[0]['timestamp_origin']==1
    assert exact[0]['coordinates']==10 and exact[0]['label_policy_full_match']['source_point_labels']


def test_unmapped_timestamp_does_not_claim_coordinate_equivalence(tmp_path):
    path=tmp_path/'csv.csv';pd.DataFrame(dict(timestamp=[100],feature_1=[1],is_anomaly=[0])).to_csv(path,index=False)
    inventory=dict(registered_CSV_files=[dict(dataset='SMD',entity='fixture',path=str(path),sha256=s.sha(path),role='test')])
    rows=s.compare_registered_CSV(dict(dataset='SMD',entity='fixture',loader='text'),dict(train=np.ones((3,1)),test=np.ones((3,1)),labels=np.zeros(3)),inventory)
    assert not rows[0]['all_original_CSV_coordinates_compared'] and rows[0]['exact_matching_candidates']==[]


def test_worker_refuses_environment_mismatch_before_writing(tmp_path):
    c=s.registration();c['environment']['numpy']='wrong';c['artifact_root']=str(tmp_path/'must_not_exist')
    with pytest.raises(ValueError,match='environment'):s.worker(c)
    assert not (tmp_path/'must_not_exist').exists()


def test_cold_full_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
