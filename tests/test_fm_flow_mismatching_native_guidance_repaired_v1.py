"""Branch parity and independent full-gradient checks; not benchmark metrics."""
from pathlib import Path

import pytest
import torch

from iia_benchmark.models.fm_native_guidance import load_verified_flowgrad,unit_time_dynamic,dflow_initial_noise_flowgrad
from iia_benchmark.models.fm_native_guidance_repaired import FlowGradPartitionRepair,load_repaired_dflow_adapter

ROOT=Path(__file__).resolve().parents[1]
AUTHOR='data/public_datasets/flow_matching/flow_mismatching_native_guidance_sources_v1/flowgrad_author_original/utils/flowgrad_utils.py'
AUTHOR_SHA='1a5348d80151004713cf27c909ab9f30b9a49045fb99cd689cdc1a6a8c13d231'
BASE='src/iia_benchmark/models/fm_native_guidance.py'
BASE_SHA='bec91d8249415b0a78dd4bfbf74f011cd1168fc649e704535ab20c3c67facbf9'


@pytest.fixture(scope='module',autouse=True)
def bounded_threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def original():return load_verified_flowgrad(AUTHOR,AUTHOR_SHA,asset_root=ROOT)
def adapter():return load_repaired_dflow_adapter(BASE,BASE_SHA,asset_root=ROOT).adapter


def test_first_boundary_wrap_repair_preserves_every_forward_state_and_call_count():
    native=original();fixed=FlowGradPartitionRepair(native);calls=[]
    def velocity(x,t):
        calls.append(t.clone());return torch.ones_like(x)*(1+t.reshape(-1,1,1,1)/999)
    z=torch.ones(2,3,2,2)
    raw,raw_partition=native.generate_traj(velocity,z,N=10,straightness_threshold=.03)
    assert len(calls)==10;calls.clear()
    repaired,repaired_partition=fixed.generate_traj(velocity,z,N=10,straightness_threshold=.03)
    assert len(calls)==10
    assert len(raw)==len(repaired)==11
    assert all(torch.equal(a,b) for a,b in zip(raw,repaired))
    assert raw_partition['length'][0]==1
    assert repaired_partition['length'][0]>1
    assert fixed.last_repair_audit['extra_forward_model_calls']==0
    assert sum(repaired_partition['length'].values())==10


def test_zero_energy_constant_field_has_finite_zero_straightness():
    native=original();fixed=FlowGradPartitionRepair(native)
    assert not torch.isfinite(torch.tensor(0.)/torch.tensor(0.))
    trajectory,partition=fixed.generate_traj(lambda x,t:torch.zeros_like(x),torch.ones(1,3,2,2),N=10,straightness_threshold=.005)
    assert partition=={'indices':[0],'length':{0:10}}
    assert fixed.last_repair_audit['per_step_straightness']==[0.]*10
    assert fixed.last_repair_audit['zero_same_velocity_ratios_defined_as_zero']==18
    assert all(torch.equal(trajectory[0],x) for x in trajectory)


def test_zero_to_nonzero_velocity_forces_boundary_without_epsilon_or_nan():
    fixed=FlowGradPartitionRepair(original())
    def velocity(x,t):return torch.zeros_like(x) if t[0]<1.1 else torch.ones_like(x)
    _,partition=fixed.generate_traj(velocity,torch.ones(1,3,2,2),N=4,straightness_threshold=.005)
    assert partition['length'][0]==1
    assert fixed.last_repair_audit['zero_velocity_nonzero_neighbor_boundaries']==1
    assert fixed.last_repair_audit['per_step_straightness'][0]=='infinity_at_zero_velocity_with_nonzero_neighbor'
    with pytest.raises(ValueError,match='finite values'):
        fixed.generate_traj(lambda x,t:torch.full_like(x,float('nan')),torch.ones(1,3,2,2),N=4,straightness_threshold=0)


def independent_full_gradient(velocity,z,target,steps):
    state=z.detach().clone().requires_grad_(True);initial=state
    dynamic=unit_time_dynamic(velocity)
    for i in range(steps):
        time=(torch.ones(z.shape[0],device=z.device)*i/steps*(1-1e-3)+1e-3)*999
        state=state+dynamic(state,time)/steps
    loss=(state-target).square().mean()
    return torch.autograd.grad(loss,initial)[0],state.detach()


def test_recorded_time_vjp_matches_independent_non_autonomous_full_gradient():
    native=original();fixed=FlowGradPartitionRepair(native)
    velocity=lambda x,t:2*t[:,None,None,None]*x+.2
    z=torch.ones(2,3,2,2,dtype=torch.float64)*.2;target=torch.ones_like(z);steps=7;lr=.1
    gradient,_=independent_full_gradient(velocity,z,target,steps)
    common=dict(steps=steps,iterations=1,learning_rate=lr,straightness_threshold=0,optimizer='sgd_diagnostic')
    raw=dflow_initial_noise_flowgrad(velocity,target,z,native_core=native,**common)
    repaired=adapter()(velocity,target,z,native_core=fixed,**common)
    expected=z-lr*gradient
    torch.testing.assert_close(repaired['optimized_initial_noise'],expected,rtol=1e-12,atol=1e-12)
    assert (raw['optimized_initial_noise']-expected).abs().max()>1e-13
    assert not repaired['optimization_trace'][0]['native_first_step_wrap_preserved']
    assert not repaired['author_anomaly_implementation_verified']


def test_repaired_lbfgs_converges_with_full_non_autonomous_trajectory():
    fixed=FlowGradPartitionRepair(original())
    target=torch.ones(1,3,2,2,dtype=torch.float64);z=torch.zeros_like(target)
    result=adapter()(lambda x,t:.3*t[:,None,None,None]*x+.1,target,z,native_core=fixed,
        steps=7,iterations=2,learning_rate=1,straightness_threshold=0,optimizer='lbfgs',lbfgs_inner_iterations=20)
    assert result['final_MSE']<1e-12
    assert result['repair_audit']['first_boundary_wrap_removed']
    assert not result['original_source_modified']


def test_repair_provenance_is_required_and_all_immutable_original_files_unchanged():
    before=[(ROOT/p).read_bytes() for p in [AUTHOR,BASE]]
    result=load_repaired_dflow_adapter(BASE,BASE_SHA,asset_root=ROOT)
    assert result.controlled_AST_patches==2 and not result.original_source_file_modified
    z=torch.ones(1,3,2,2)
    with pytest.raises(ValueError,match='repaired native core'):
        result.adapter(lambda x,t:x,z,z,native_core=original(),steps=2,iterations=1,learning_rate=1,straightness_threshold=0,optimizer='sgd_diagnostic')
    with pytest.raises(ValueError,match='SHA256'):
        load_repaired_dflow_adapter(BASE,'0'*64,asset_root=ROOT)
    assert before==[(ROOT/p).read_bytes() for p in [AUTHOR,BASE]]
