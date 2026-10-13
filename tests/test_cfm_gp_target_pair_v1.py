import pytest
from scripts.flow_matching.audit_cfm_gp_target_pair_v1 import paired_summary


def test_group_bootstrap_keeps_pairing_and_conditions_on_trained_models():
    result=paired_summary([1.,2.],[2.,3.],[10,11],7,200)
    assert result['paired_MSE_difference_conditional_minus_literal']==1.
    assert result['paired_trajectory_bootstrap_ci95']==[1.,1.]
    assert result['bootstrap_unit']=='whole held-out trajectory'
    assert not result['training_seed_uncertainty_estimated']


def test_cells_and_duplicate_trajectories_cannot_be_treated_as_independent_groups():
    for a,b,ids in [([[1.,2.]],[[1.,2.]],[1]),([1.,2.],[2.,3.],[1,1]),([1.],[2.,3.],[1])]:
        with pytest.raises(ValueError):paired_summary(a,b,ids,7,200)
