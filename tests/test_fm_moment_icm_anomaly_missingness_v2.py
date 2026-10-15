import numpy as np
import pytest
from scripts.flow_matching import audit_moment_icm_anomaly_missingness_v2 as a


@pytest.mark.parametrize('mask,expected',[
    ([],[]),([False,False],[]),([True,True],[[0,2]]),
    ([True,False,True,True,False,True],[[0,1],[2,4],[5,6]])])
def test_all_nonfinite_positions_are_recovered_without_fill_or_drop(mask,expected):
    mask=np.asarray(mask,dtype=bool);ranges=a.missing_ranges(mask)
    assert ranges==expected
    actual=np.zeros(len(mask),dtype=bool)
    for lo,hi in ranges:actual[lo:hi]=True
    assert np.array_equal(actual,mask)
