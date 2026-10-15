import io,struct
import pytest
from scripts.flow_matching import audit_full873_window_geometry_v1 as s


@pytest.mark.parametrize('version',[(1,0),(2,0),(3,0)])
def test_real_NPY_header_versions_without_reading_payload(version):
    raw=repr(dict(descr='<f4',fortran_order=False,shape=(100,25))).encode()
    prefix=b'\x93NUMPY'+bytes(version)+struct.pack('<H' if version==(1,0) else '<I',len(raw))
    stream=io.BytesIO(prefix+raw+b'payload_must_not_be_read')
    assert s.array_header(stream)['shape']==[100,25]
    assert stream.read()==b'payload_must_not_be_read'


def inputs(n=100):
    dataset=dict(entities=[dict(entity_id='short_real_geometry_fixture',array_prefix='e',features=25,
        fit_interval=[0,112],validation_interval=[212,212+n],test_points=200)])
    headers={key:dict(shape=shape) for key,shape in [('e_train',[112,25]),('e_validation',[n,25]),('e_test',[200,25])]}
    return dataset,headers


@pytest.mark.parametrize('window,training,late',[(100,False,False),(105,False,True),(512,True,False)])
def test_exactly100_validation_points_preserve_true105_and512_rejections(window,training,late):
    dataset,headers=inputs();record=s.inspect_case(dict(window=window),dataset,headers)
    assert record['training_rejected_before_fit']==training
    assert record['possible_later_scoring_rejection_after_fit']==late
    assert record['all_entities_pass_fixed_window_length_gate']==(window==100)
    assert not record['actual_job_terminal_failure_observed']


def test_manifest_intervals_must_match_actual_NPZ_header():
    dataset,headers=inputs();headers['e_validation']['shape'][0]=105
    with pytest.raises(ValueError):s.inspect_case(dict(window=105),dataset,headers)
