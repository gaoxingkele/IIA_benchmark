import numpy as np
import pytest
from scripts.flow_matching import audit_NASA_split_representation_v1 as s


def test_all_six_files_and_fixed96comparisons_are_registered():
    c=s.registration();s.verify(c)
    assert c['required_entities']==2 and c['required_CSV_files']==6
    assert len(c['profiles'])*len(c['raw_roles'])*len(c['timestamp_origins'])*6==96
    assert all(set(j['CSV_roles'])=={'train','val','test'} for j in c['jobs'])
    assert not c['numpy_tolerance_used'] and not c['labels_used_to_select_representation']


@pytest.mark.parametrize('field',['jobs','profiles','timestamp_origins','settings','numpy_tolerance_used'])
def test_reduced_hypotheses_or_changed_tolerance_and_resources_rejected(field):
    c=s.registration()
    if field in ['jobs','profiles','timestamp_origins']:c[field].pop()
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=1
    else:c[field]=True
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('profile',s.PROFILES)
def test_declared_transform_has_same_geometry_and_does_not_modify_raw(profile):
    raw=np.array([[.1,.12345678901234567],[-.3816289182780123,1.],[0.,-1.]])
    before=raw.copy();result=s.representation(raw,profile)
    assert result.shape==raw.shape and result.dtype==np.float64 and np.array_equal(raw,before)
    assert np.isfinite(result).all()
    if profile=='direct_float64':assert np.array_equal(result,raw)
    if profile=='float32_then_float64':assert np.array_equal(result,raw.astype(np.float32).astype(np.float64))


def test_no_unregistered_profile_is_implicitly_tried():
    with pytest.raises(ValueError):s.representation(np.ones((3,1)),'tune_until_equal')


def test_exact_coordinate_contract_rejects_even_one_different_bit():
    raw=np.arange(8,dtype=float).reshape(4,2);values=raw[1:3].copy();times=np.array([2,3])
    result,rows=s.compare(values,times,raw,1)
    assert result['every_coordinate_equal'] and rows.tolist()==[1,2]
    values[0,0]=np.nextafter(values[0,0],np.inf)
    result,_=s.compare(values,times,raw,1)
    assert not result['every_coordinate_equal'] and result['unequal_coordinates']==1


def test_out_of_bounds_candidate_is_recorded_without_omitting_rows():
    result,rows=s.compare(np.ones((2,1)),np.array([4,5]),np.ones((4,1)),0)
    assert not result['in_bounds'] and rows is None and result['compared_coordinates']==0


def test_half_split_is_a_partition_only_when_every_original_row_occurs_once():
    good=s.partition(np.arange(4),np.arange(4,9),9)
    assert good['every_original_test_row_once'] and good['validation_precedes_test'] and good['overlap_rows']==0
    assert not s.partition(np.arange(4),np.arange(3,9),9)['every_original_test_row_once']
    assert not s.partition(np.arange(3),np.arange(4,9),9)['every_original_test_row_once']
