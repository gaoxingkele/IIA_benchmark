import copy
import pytest
import numpy as np

from scripts.flow_matching.audit_spectral_full_trainer_sampling_graph_v1 import upstream_contract,full_trajectory_contract,bitwise_array_equal

def test_full_native_bitwise_comparison_rejects_signed_zero_and_dtype_changes():
    data=np.asarray([0.,1.],dtype='float32')
    assert bitwise_array_equal(data,data.copy())
    assert not bitwise_array_equal(data,np.asarray([-0.,1.],dtype='float32'))
    assert not bitwise_array_equal(data,data.astype('float64'))
    assert not bitwise_array_equal(data,data.reshape(2,1))

def native_operator_evidence():
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_derivative_graph_audit_passed=True)
    proof=dict(passed=True,actual_model_parameters=356800,full_batch_shape=[2001,24,6],full_cases=5,
        all_full_first_derivatives_bitwise_equal=True,complete_native_EMA_checkpoint_restored_bitwise=True,
        native_EMA_detached_parameter_flags_restored=True,original_training_higher_order_forward_restored=True,
        cases=[dict(time=t,shape=[2001,24,6],full_field_elements=288144,bitwise_derivatives_equal=True,
            Torch_RNG_consumption_unchanged=True,all_full_native_parameters_unchanged=True) for t in [0.,.25,.5,.75,.99999]])
    return receipt,proof

def test_full_trajectory_audit_requires_actual_full_native_operator_success():
    receipt,proof=native_operator_evidence()
    upstream_contract(receipt,proof)
    for key,value in [('exit_code',1),('resource_guard_aborted',True),('full_native_derivative_graph_audit_passed',False)]:
        with pytest.raises(ValueError):upstream_contract(dict(receipt,**{key:value}),proof)
    for key,value in [('passed',False),('actual_model_parameters',100),('full_batch_shape',[128,24,6]),
        ('full_cases',4),('all_full_first_derivatives_bitwise_equal',False),('native_EMA_detached_parameter_flags_restored',False)]:
        with pytest.raises(ValueError):upstream_contract(receipt,dict(proof,**{key:value}))

def test_every_original_native_operator_case_is_required_before_integration():
    receipt,proof=native_operator_evidence()
    for key,value in [('shape',[128,24,6]),('full_field_elements',18432),('Torch_RNG_consumption_unchanged',False),
        ('bitwise_derivatives_equal',False),('all_full_native_parameters_unchanged',False),('time',.2)]:
        bad=copy.deepcopy(proof)
        bad['cases'][2][key]=value
        with pytest.raises(ValueError):upstream_contract(receipt,bad)
    with pytest.raises(ValueError):upstream_contract(receipt,dict(proof,cases=proof['cases'][:-1]))

def test_complete_native_two_cycle_sampling_must_not_be_replaced_by_one_batch_or_shorter_grid():
    config=dict(full_generation_cycles=2,full_field_calls_per_pass=1000,full_generated_shape=[4002,24,6])
    full_trajectory_contract(config,2,1000,[4002,24,6])
    for cycles,calls,shape in [(1,500,[2001,24,6]),(2,998,[4002,24,6]),(2,1000,[3662,24,6]),(2,1000,[4002,12,6])]:
        with pytest.raises(ValueError):full_trajectory_contract(config,cycles,calls,shape)
