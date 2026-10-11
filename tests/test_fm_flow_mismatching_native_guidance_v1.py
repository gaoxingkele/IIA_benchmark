"""Original VJP on artificial fields and analytic source-gradient comparisons."""
import hashlib
from pathlib import Path

import pytest
import torch

from iia_benchmark.models.fm_native_guidance import load_verified_flowgrad,unit_time_dynamic,dflow_initial_noise_flowgrad

ROOT=Path(__file__).resolve().parents[1]
SOURCE='data/public_datasets/flow_matching/flow_mismatching_native_guidance_sources_v1/flowgrad_author_original/utils/flowgrad_utils.py'
SHA='1a5348d80151004713cf27c909ab9f30b9a49045fb99cd689cdc1a6a8c13d231'


@pytest.fixture(scope='module',autouse=True)
def bounded_threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def core():
    assert (ROOT/SOURCE).is_file(), 'Original frozen FlowGrad source is required, never silently skip'
    return load_verified_flowgrad(SOURCE,SHA,asset_root=ROOT)


def test_original_source_loader_is_immutable_and_rejects_unknown_bytes(tmp_path):
    before=(ROOT/SOURCE).read_bytes();native=core()
    assert callable(native.generate_traj) and callable(native.flowgrad_optimization)
    assert (ROOT/SOURCE).read_bytes()==before
    changed=tmp_path/'changed.py';changed.write_bytes(before+b'\n# changed original\n')
    with pytest.raises(ValueError,match='SHA256'):
        load_verified_flowgrad(changed,SHA)


def test_time_argument_adaptation_preserves_fixed_labels_and_native_epsilon():
    observations=[]
    def velocity(x,t,labels):observations.append((t.clone(),labels.clone()));return x
    dynamic=unit_time_dynamic(velocity,torch.tensor([2,4]))
    x=torch.ones(2,3,2,2)
    dynamic(x,torch.tensor([.001*999]))
    torch.testing.assert_close(observations[0][0],torch.full((2,),.001))
    assert torch.equal(observations[0][1],torch.tensor([2,4]))


def test_original_full_partition_controls_match_independent_full_autograd(capsys):
    native=core();steps=4;z=torch.ones(1,1,1,2,dtype=torch.float64)
    dynamic=lambda x,t:.2*x+.1
    target=torch.zeros_like(z);lr=.1
    # Native u optimization is independently differentiated through every step.
    controls=[torch.zeros_like(z,requires_grad=True) for _ in range(steps)]
    state=z
    for control in controls:
        state=state+control
        state=state+dynamic(state,None)/steps
    gradients=torch.autograd.grad((state-target).square().mean(),controls)
    observed=native.flowgrad_optimization(z,list(range(steps)),dynamic,native.generate_traj,
        N=steps,L_N=lambda x:(x-target).square().mean(),number_of_iterations=1,
        straightness_threshold=0,lr=lr)
    for j,g in enumerate(gradients):torch.testing.assert_close(observed[j],-lr*g,rtol=1e-12,atol=1e-12)
    assert 'iteration:' in capsys.readouterr().out


def test_source_space_adapter_matches_closed_form_initial_gradient_under_no_grad():
    native=core();steps=5;lr=.05
    coefficient=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    x=torch.ones(2,3,2,2,dtype=torch.float64)*.7
    z=torch.ones_like(x)*.4
    velocity=lambda y,t:coefficient*y+.2
    factor=(1+.3/steps)**steps
    offset=.2/.3*(factor-1)
    endpoint=factor*z+offset
    gradient=2*(endpoint-x)*factor/x.numel()
    with torch.no_grad():
        result=dflow_initial_noise_flowgrad(velocity,x,z,native_core=native,steps=steps,
            iterations=1,learning_rate=lr,straightness_threshold=0,optimizer='sgd_diagnostic')
    torch.testing.assert_close(result['optimized_initial_noise'],z-lr*gradient,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(result['reconstruction'],factor*(z-lr*gradient)+offset,rtol=1e-12,atol=1e-12)
    assert coefficient.grad is None
    assert result['optimization_trace'][0]['selected_indices']==list(range(steps))
    assert result['final_MSE']<float((endpoint-x).square().mean())
    assert not result['author_anomaly_implementation_verified']


def test_nonuniform_original_partition_is_preserved_for_constant_nonzero_field():
    native=core();x=torch.ones(1,3,2,2);z=torch.zeros_like(x)
    result=dflow_initial_noise_flowgrad(lambda y,t:torch.ones_like(y)*.2,x,z,
        native_core=native,steps=10,iterations=1,learning_rate=.1,straightness_threshold=.005,optimizer='sgd_diagnostic')
    assert result['optimization_trace'][0]['selected_indices']==[0]
    assert result['optimization_trace'][0]['intervals']=={0:10}
    assert result['optimization_trace'][0]['native_first_step_wrap_preserved']
    assert result['heatmap'].shape==(1,2,2)


def test_unreported_compute_arguments_cannot_be_invented_or_implicitly_defaulted():
    native=core();x=torch.ones(1,3,2,2)
    with pytest.raises(TypeError):
        dflow_initial_noise_flowgrad(lambda x,t:x,x,x,native_core=native)
    with pytest.raises(ValueError,match='budgets'):
        dflow_initial_noise_flowgrad(lambda x,t:x,x,x,native_core=native,steps=0,iterations=1,learning_rate=1,straightness_threshold=0,optimizer='sgd_diagnostic')
    with pytest.raises(ValueError,match='inner budget'):
        dflow_initial_noise_flowgrad(lambda x,t:x,x,x,native_core=native,steps=2,iterations=1,learning_rate=1,straightness_threshold=0,optimizer='lbfgs')


def test_dflow_lbfgs_source_noise_line_search_converges_on_independent_linear_case():
    native=core();x=torch.ones(1,3,2,2,dtype=torch.float64)
    coefficient=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    z=torch.zeros_like(x)
    result=dflow_initial_noise_flowgrad(lambda y,t:coefficient*y+.1,x,z,
        native_core=native,steps=5,iterations=2,learning_rate=1.,straightness_threshold=0,
        optimizer='lbfgs',lbfgs_inner_iterations=20)
    assert result['final_MSE']<1e-12
    assert result['optimizer']=='lbfgs' and result['line_search']=='strong_wolfe_local_choice'
    assert coefficient.grad is None
    assert result['closure_evaluations']==len(result['optimization_trace'])
    factor=(1+.2/5)**5;offset=.1/.2*(factor-1)
    torch.testing.assert_close(result['optimized_initial_noise'],(x-offset)/factor,atol=1e-6,rtol=1e-6)
