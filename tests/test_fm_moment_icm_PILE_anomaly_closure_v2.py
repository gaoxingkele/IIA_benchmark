import pytest
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v2 as v


@pytest.mark.parametrize('corruption',['none','overlap','missing_original','missing_repair','extra_repair','duplicate_job'])
def test_full1949_scope_keeps_exact_disjoint_original_and_repair_ownership(corruption):
    jobs=[dict(id=str(i)) for i in range(1949)]
    original={str(i):{} for i in range(1627)};repair={str(i):{} for i in range(1627,1949)}
    baseline=v.choose_records(jobs,original,repair)
    assert len(baseline)==1949 and sum(track=='dependency_repair' for _,track in baseline)==322
    if corruption=='none':return
    if corruption=='overlap':repair['0']={}
    elif corruption=='missing_original':del original['0']
    elif corruption=='missing_repair':del repair['1948']
    elif corruption=='extra_repair':repair['extra']={}
    elif corruption=='duplicate_job':jobs[-1]=dict(id='0')
    with pytest.raises(ValueError):v.choose_records(jobs,original,repair)
