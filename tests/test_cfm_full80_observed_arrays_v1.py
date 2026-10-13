from scripts.flow_matching.audit_cfm_full80_observed_arrays_v1 import tail_evidence


def test_finite_small_velocity_cannot_be_claimed_float32_overflow():
    row=dict(trajectory_id=2,RHS_calls=2,maximum_finite_state_abs=1.,maximum_finite_velocity_abs=.5,
             last_RHS_observations=[dict(call=1,state_finite=True,velocity_finite=True),
                                    dict(call=2,state_finite=True,velocity_finite=True)])
    evidence=tail_evidence(row,3.4e38)
    assert not evidence['finite_velocity_near_float32_limit']
    assert not evidence['nonfinite_velocity_observed']
    assert not evidence['finite_velocity_then_nonfinite']
    assert not evidence['learned_training_failure_root_cause_certified']


def test_recorded_nonfinite_velocity_precedes_nonfinite_state():
    row=dict(trajectory_id=103,RHS_calls=4,maximum_finite_state_abs=7.2e32,maximum_finite_velocity_abs=3.19e38,
             last_RHS_observations=[dict(call=1,state_finite=True,velocity_finite=True),
                                    dict(call=2,state_finite=True,velocity_finite=False),
                                    dict(call=3,state_finite=False,velocity_finite=False)])
    evidence=tail_evidence(row,3.4e38)
    assert evidence['finite_velocity_near_float32_limit']
    assert evidence['finite_state_at_first_nonfinite_velocity']
    assert evidence['first_nonfinite_velocity_tail_observation']['call']==2
    assert evidence['first_nonfinite_state_tail_observation']['call']==3
    assert not evidence['learned_training_failure_root_cause_certified']
