import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_PILE_long_forecast_v1 as s


@pytest.mark.parametrize('name,count', [('ETTh1',17420),('ETTm1',69680),('weather',52696),('national_illness',966)])
def test_source_time_borders_and_historical_context_overlap(name,count):
    b=s.expected_borders(name,count)
    assert b['train'][0]==0 and s.overlap(b['train'],b['val'])==256
    assert b['test'][1]<=count
    if name=='national_illness':assert s.overlap(b['train'],b['test'])==158
    if name=='ETTh1':assert b['test'][1]==14400
    if name=='ETTm1':assert b['test'][1]==57600


def test_independent_row_origin_rejects_validation_reset_to_zero(tmp_path):
    p=tmp_path/'rows.npy';np.save(p,np.arange(100,120,dtype=np.int64));s.check_rows(p,100,20)
    with pytest.raises(ValueError):s.check_rows(p,0,20)


def test_separate_OT_comparison_keeps_exact_differences_without_clipping(tmp_path):
    a=np.arange(9,dtype=np.float64).reshape(9,1);b=np.column_stack([a[:,0]+5,a[:,0]])
    pa,pb=tmp_path/'a.npy',tmp_path/'b.npy';np.save(pa,a);np.save(pb,b)
    assert s.compare_OT(pa,pb,1)['exactly_different_coordinates']==0
    b[-1,1]+=.25;np.save(pb,b)
    r=s.compare_OT(pa,pb,1)
    assert r['exactly_different_coordinates']==1 and r['maximum_absolute_difference']==.25
    assert not r['tolerance_applied'] and r['separate_input_tracks_not_pooled']
