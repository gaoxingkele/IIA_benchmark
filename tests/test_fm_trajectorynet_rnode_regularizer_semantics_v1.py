import copy
import pytest
from scripts.flow_matching import audit_trajectorynet_rnode_regularizer_semantics_v1 as s

@pytest.mark.parametrize('dim',[2,5,25])
def test_pinned_original_energy_and_Jacobian_values_and_gradients(dim):
    c=s.registration();s.verify(c);r=s.compare(c,dim)
    assert r['energy_value_and_gradient_ratio']==2*dim
    assert r['original_RNODE_context_tensor_identity_preserved']
    assert not r['trajectorynet_Jacobian_is_squared_Frobenius']
    assert r['optimizer_updates']==0 and r['analytic_fixture_not_dataset_experiment']

def test_unknown_EB_coefficient_and_full_training_remain_unresolved():
    c=s.registration();m=s.read(s.ROOT/s.MODEL)
    assert m['EB_Jacobian_coefficient'] is None
    assert not m['full_original_energy_grid_training_complete']
    assert not c['EB_Jacobian_coefficient_certified'] and not c['benchmark_performance_claim']
    changed=copy.deepcopy(c);changed['full_fixture_batch']=100
    with pytest.raises(ValueError,match='Pinned original'):s.verify(changed)
