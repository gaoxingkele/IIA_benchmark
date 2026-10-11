import copy
import numpy as np
import pytest
from scripts.flow_matching.audit_saits_physio_cpu_cuda_full_pairs_v1 import compare_pair,group_statistics

TOLERANCE=dict(rtol=2e-6,atol=2e-6)


def pair():
    shape=[4,2,2];truth=np.ones(shape);mask=np.ones(shape);units=np.array(['a','b','c','d'])
    predictions=truth+.25;ids,stats=group_statistics(predictions,truth,mask,units,np)
    cpu=dict(predictions=predictions,truth=truth,mask=mask,test_indices=np.arange(4),unit_ids=units,group_ids=ids,group_statistics=stats)
    cuda=copy.deepcopy(cpu);cuda['predictions'][0,0,0]+=.001
    cuda['group_ids'],cuda['group_statistics']=group_statistics(cuda['predictions'],truth,mask,units,np)
    original=dict(unit_ids=units,group_ids=ids,statistics=stats)
    return cpu,cuda,original,shape


def test_full_pair_comparison_records_difference_and_preserves_original_strict_failure():
    cpu,cuda,original,shape=pair();report,artifacts=compare_pair(cpu,cuda,original,TOLERANCE,shape,16)
    assert report['full_prediction_coordinates']==16 and report['held_out_prediction_coordinates_different']==1
    assert report['original_strict_statistics_reviews']['cpu']['strict_original_statistics_match']
    assert not report['original_strict_statistics_reviews']['cuda']['strict_original_statistics_match']
    assert report['original_strict_statistics_reviews']['cuda']['failed_patients']==1
    assert report['original_numerical_mismatch_cause_proven'] is False and report['full_native_reproduction_certified'] is False
    assert np.array_equal(artifacts['original_statistics'],original['statistics'])


def test_identical_full_arrays_do_not_prove_original_training_prediction_identity():
    cpu,_,original,shape=pair();report,_=compare_pair(cpu,copy.deepcopy(cpu),original,TOLERANCE,shape,16)
    assert report['prediction_arrays_bitwise_equal']
    assert report['original_training_predictions_persisted'] is False
    assert report['full_native_reproduction_certified'] is False


def test_signed_zero_is_not_claimed_bitwise_identical():
    cpu,_,original,shape=pair()
    cpu['predictions'][0,0,0]=0.
    cpu['group_ids'],cpu['group_statistics']=group_statistics(cpu['predictions'],cpu['truth'],cpu['mask'],cpu['unit_ids'],np)
    original['statistics']=cpu['group_statistics']
    cuda=copy.deepcopy(cpu);cuda['predictions'][0,0,0]=-0.
    report,_=compare_pair(cpu,cuda,original,TOLERANCE,shape,16)
    assert report['full_prediction_coordinates_different']==0
    assert report['prediction_arrays_bitwise_equal'] is False


@pytest.mark.parametrize('change',['key','shape','truth','mask','indices','units','group_ids','self_statistics','original_ids','original_statistics','nonfinite'])
def test_partial_or_misaligned_complete_arrays_cannot_produce_a_comparison(change):
    cpu,cuda,original,shape=pair()
    if change=='key':cuda.pop('group_statistics')
    elif change=='shape':cuda['predictions']=cuda['predictions'][:-1]
    elif change=='truth':cuda['truth'][0,0,0]=2
    elif change=='mask':cuda['mask'][0,0,0]=0
    elif change=='indices':cuda['test_indices']=cuda['test_indices'][::-1]
    elif change=='units':cuda['unit_ids']=cuda['unit_ids'][::-1]
    elif change=='group_ids':cuda['group_ids']=cuda['group_ids'][::-1]
    elif change=='self_statistics':cuda['group_statistics'][0,1]+=1
    elif change=='original_ids':original['group_ids']=original['group_ids'][::-1]
    elif change=='original_statistics':original['statistics']=original['statistics'][:-1]
    elif change=='nonfinite':cuda['predictions'][0,0,0]=np.nan
    with pytest.raises(ValueError):compare_pair(cpu,cuda,original,TOLERANCE,shape,16)


def test_changed_mask_coverage_cannot_be_hidden_as_a_full_pair():
    cpu,cuda,original,shape=pair()
    with pytest.raises(ValueError):compare_pair(cpu,cuda,original,TOLERANCE,shape,15)
