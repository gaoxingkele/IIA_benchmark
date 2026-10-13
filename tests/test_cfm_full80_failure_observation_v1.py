import pytest
from scripts.flow_matching import audit_cfm_sine600_full80_failure_observation_v1 as current


def test_one_failed_original_trajectory_prevents_any_complete_or_censored_metric_claim():
    records=[dict(trajectory_id=i,finite_full_grid_completed=i!=22) for i in range(80)]
    result=current.validity_summary(records,80)
    assert result['attempted_original_trajectories']==80 and result['finite_full_grid_trajectories']==79
    assert result['failed_trajectory_ids']==[22]
    assert result['full_original_MSE'] is None and not result['original_scoring_certified_complete']


def test_partial_or_duplicate_coverage_rejected():
    with pytest.raises(ValueError):current.validity_summary([dict(trajectory_id=1,finite_full_grid_completed=True)]*80,80)


def test_observer_preserves_native_model_outputs_and_reports_nonfinite_velocities():
    import torch
    class Field(torch.nn.Module):
        def forward(self,t,x):return x*2
    model=Field();x=torch.tensor([3.]);t=torch.tensor(0.)
    expected=model(t,x);observer=current.RHSObserver(torch);handle=model.register_forward_hook(observer)
    assert torch.equal(model(t,x),expected)
    model(t,torch.tensor([float('inf')]))
    handle.remove()
    assert observer.calls==2 and observer.nonfinite_velocity_seen
    assert observer.maximum_finite_velocity==6.
