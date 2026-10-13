import pytest
from scripts.flow_matching.audit_commit_capacity_plan_v1 import projection,GIB


def test_extra_capacity_projection_does_not_lower_frozen_requirement():
    p=projection(68*GIB,41*GIB,32*GIB,56*GIB)
    assert p['current_available_bytes']==27*GIB and not p['current_admission_met']
    assert p['proposed_available_bytes']==59*GIB and p['proposed_admission_met']
    assert p['proposal_is_not_applied']


def test_exact_threshold_passes_but_insufficient_extension_remains_pending():
    assert projection(68*GIB,41*GIB,13*GIB,40*GIB)['proposed_admission_met']
    assert not projection(68*GIB,41*GIB,16*GIB,56*GIB)['proposed_admission_met']
    with pytest.raises(ValueError):projection(1,2,0,1)
