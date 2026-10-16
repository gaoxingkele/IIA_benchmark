from copy import deepcopy
import numpy as np
import pytest
from scripts.flow_matching import audit_industrial_full_raw_preprocessing_v1 as s


def groups():
    return {role:[dict(source_path=role,group_id=role,source_interval=[0,3],values=np.array(values,dtype=float))]
        for role,values in [('train',[[1,5],[2,5],[3,5]]),('validation',[[11,6],[12,7],[13,8]]),('test',[[21,9],[22,10],[23,11]])]}


def test_fit_uses_only_train_and_keeps_constant_channel_rule():
    a=groups();out,scaler=s.independently_standardize(a)
    b=deepcopy(a);b['test'][0]['values']*=1e6;b['validation'][0]['values']*=1e3
    _,second=s.independently_standardize(b)
    assert scaler==second and scaler['mean']==[2.,5.] and scaler['constant_channels']==[1]
    assert out['test'][0][:,1].tolist()==[4.,5.,6.]


def test_nonfinite_repair_uses_training_median_and_does_not_edit_raw_arrays():
    a=groups();a['train'][0]['values'][1,0]=np.nan;a['test'][0]['values'][0,0]=np.inf
    original=deepcopy(a);out,scaler=s.independently_standardize(a)
    assert scaler['median']==[2.,5.] and scaler['repairs']['train']==[1] and scaler['repairs']['test']==[1]
    assert out['test'][0][0,0]==0
    assert np.array_equal(a['test'][0]['values'],original['test'][0]['values'],equal_nan=True)


@pytest.mark.parametrize('fault',['source_overlap','duplicate_interval','cross_group','missing_role','feature_mismatch'])
def test_leakage_repeated_intervals_or_incomplete_roles_are_rejected(fault):
    a=groups()
    if fault=='source_overlap':a['test'][0]['source_path']='train'
    elif fault=='duplicate_interval':a['train'].append(deepcopy(a['train'][0]))
    elif fault=='cross_group':a['test'][0]['group_id']='train'
    elif fault=='missing_role':a['validation']=[]
    else:a['test'][0]['values']=a['test'][0]['values'][:,:1]
    with pytest.raises(ValueError):s.independently_standardize(a)


@pytest.mark.parametrize('fault',['last_value','dtype','shape','last_label'])
def test_full_payload_checks_every_tail_and_label_byte(fault):
    a=np.arange(15,dtype=np.float32).reshape(5,3);b=a.copy()
    if fault=='last_value':b[-1,-1]+=1
    elif fault=='dtype':b=b.astype(np.float64)
    elif fault=='shape':b=b.reshape(3,5)
    else:a=np.array([0,1,1,0],dtype=np.uint8);b=np.array([0,1,1,1],dtype=np.uint8)
    with pytest.raises(ValueError):s.exact_payload('complete',a,b)


def test_pronto_normal_selection_never_crosses_fault_gaps():
    labels=['Normal','Normal','FaultA','Normal','FaultB','Normal','Normal']
    assert s.normal_spans(labels)==[(0,2),(3,4),(5,7)]
    assert s.normal_spans(['FaultA'])==[]
    assert s.normal_spans(['Normal'])==[(0,1)]


def test_reconstructed_payload_includes_dtype_shape_and_all_coordinates():
    a=np.array([[1,2],[3,4]],dtype=np.float32)
    proof=s.exact_payload('whole',a,a.copy())
    assert proof['shape']==[2,2] and proof['elements']==4 and len(proof['sha256'])==64
