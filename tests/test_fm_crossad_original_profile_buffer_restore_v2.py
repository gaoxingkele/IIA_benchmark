import pytest
import torch

from scripts.flow_matching.replay_crossad_full_native_model_v2 import restore_original_profile_buffers


def model():return torch.nn.Sequential(torch.nn.Linear(2,1))


def test_restore_only_original_metadata_then_load_every_algorithm_tensor_strictly():
    network=model();state=network.state_dict()
    state['total_ops']=torch.tensor([12.],dtype=torch.float64)
    state['0.total_params']=torch.tensor([3.],dtype=torch.float64)
    records=restore_original_profile_buffers(network,state)
    assert len(records)==2
    network.load_state_dict(state,strict=True)
    assert network.total_ops.dtype==torch.float64
    assert len(restore_original_profile_buffers(network,state))==0
    assert sum(p.numel() for p in network.parameters())==3


@pytest.mark.parametrize('name,value',[('extra.weight',torch.ones(1)),('1.total_ops',torch.ones(1)),
    ('total_params',torch.ones(2)),('total_ops',torch.tensor([float('nan')]))])
def test_unknown_algorithm_weights_modules_and_invalid_metadata_are_rejected(name,value):
    network=model();state=network.state_dict();state[name]=value
    with pytest.raises(ValueError):restore_original_profile_buffers(network,state)


def test_a_missing_original_algorithm_parameter_is_never_accepted():
    network=model();state=network.state_dict();state.pop('0.weight')
    with pytest.raises(ValueError):restore_original_profile_buffers(network,state)
