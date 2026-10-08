"""Mathematical and executable-contract tests, not benchmark performance."""
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_native import (
    GraphSpectralPath,GRASPDetector,DFMDetector,PrismFlowCore,PrismFlowModel,
    normalized_laplacian,dual_cosine_loss,cnf_log_density,
)


def test_graph_path_endpoints_derivative_and_zero_frequency():
    lap=normalized_laplacian(torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=torch.double))
    path=GraphSpectralPath(lap)
    x0=torch.randn(2,3,4,dtype=torch.double);x1=torch.randn_like(x0)
    torch.testing.assert_close(path(x0,x1,torch.zeros(2))[0],x0,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(path(x0,x1,torch.ones(2))[0],x1,atol=1e-12,rtol=1e-12)
    t=torch.tensor([0.3,0.6],dtype=torch.double);h=1e-5
    numerical=(path(x0,x1,t+h)[0]-path(x0,x1,t-h)[0])/(2*h)
    torch.testing.assert_close(numerical,path(x0,x1,t)[1],atol=1e-8,rtol=1e-8)
    linear=GraphSpectralPath(lap,tau=0.)
    state,velocity=linear(x0,x1,t)
    torch.testing.assert_close(state,(1-t[:,None,None])*x0+t[:,None,None]*x1)
    torch.testing.assert_close(velocity,x1-x0)
    torch.testing.assert_close(linear.weights(t).expand(-1,3,-1),t[:,None,None].square().expand(-1,3,-1))


def test_graph_basis_rotation_invariance():
    # Complete graph has repeated high-frequency eigenvalues.
    lap=normalized_laplacian(torch.ones(3,3,dtype=torch.double)-torch.eye(3,dtype=torch.double))
    path=GraphSpectralPath(lap);rotated=GraphSpectralPath(lap)
    angle=0.37
    rotation=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=torch.double)
    rotated.basis[:,1:]=rotated.basis[:,1:]@rotation
    source=torch.randn(2,3,4,dtype=torch.double);target=torch.randn_like(source);t=torch.tensor([0.2,0.8])
    for a,b in zip(path(source,target,t),rotated(source,target,t)):torch.testing.assert_close(a,b)
    residual=torch.randn_like(source)
    score=lambda p:(p.spectral(residual).square()*p.weights(t)).sum((1,2))
    torch.testing.assert_close(score(path),score(rotated))


@pytest.mark.parametrize("variant",["grasp","mean","er","lin"])
def test_grasp_variants_train_score_and_do_not_mutate_input(variant):
    x=np.random.default_rng(11).normal(size=(8,4,3)).astype(np.float32);saved=x.copy()
    adjacency=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
    detector=GRASPDetector(window=4,hidden=8,epochs=1,batch_size=4,dropout=0.,
                           source_samples=2,flow_evaluations=2,variant=variant,device="cpu").fit(x,adjacency)
    score=detector.score(x)
    assert score.shape==(8,) and np.isfinite(score).all() and (score>=0).all()
    np.testing.assert_array_equal(x,saved)
    np.testing.assert_allclose(score,detector.score(x))
    detector.batch_size=3
    np.testing.assert_allclose(score,detector.score(x),rtol=1e-5,atol=1e-5)
    assert np.count_nonzero(detector.adjacency_)==4
    if variant=="lin":assert torch.equal(detector.path.omega,torch.zeros(3))


def test_dfm_literal_cosine_degeneracy_is_visible():
    # Equation 12 has zero loss for aligned constant fields, regardless of data.
    # This test prevents claiming bijectivity merely because training loss drops.
    forward=torch.tensor([[2.,3.],[2.,3.]])
    reverse=forward*4
    assert dual_cosine_loss(forward,reverse)<1e-6
    assert dual_cosine_loss(forward,-reverse)>1.99


@pytest.mark.parametrize("trace",["exact","hutchinson"])
def test_cnf_density_sign_for_linear_field(trace):
    x=torch.tensor([[0.1,0.2],[1.,2.]],dtype=torch.double)
    rate=0.2
    field=lambda time,state:rate*state
    steps=1000
    log_density=cnf_log_density(field,x,steps=steps,trace=trace)
    z=x*np.exp(-rate)
    expected=-0.5*(z.square()+np.log(2*np.pi)).sum(-1)-rate*x.shape[1]
    torch.testing.assert_close(log_density,expected,rtol=2e-4,atol=2e-4)


def test_dfm_train_score_contract():
    x=np.random.default_rng(3).normal(size=(6,3,2)).astype(np.float32)
    detector=DFMDetector(window=3,hidden=8,epochs=2,batch_size=3,trace="exact",device="cpu").fit(x)
    scores=detector.score(x)
    assert len(detector.loss_history_)==2
    assert scores.shape==(6,) and np.isfinite(scores).all()


def test_cnf_translation_has_no_density_correction():
    x=torch.tensor([[1.,2.]],dtype=torch.double)
    logp=cnf_log_density(lambda time,state:torch.ones_like(state),x,steps=5)
    expected=-0.5*((x-1).square()+np.log(2*np.pi)).sum(-1)
    torch.testing.assert_close(logp,expected)


def test_prism_generator_is_dissipative():
    core=PrismFlowCore(4,latent=3,experts=3,delta=0.2)
    with torch.no_grad():core.S.normal_();core.R.normal_()
    a=core.generators(); symmetric=(a+a.transpose(-1,-2))/2
    assert torch.linalg.eigvalsh(symmetric).max()<=-0.2+1e-5


def test_prism_wta_masks_nonwinning_generator_gradients():
    core=PrismFlowCore(4,hidden=8,latent=3,experts=3,alpha_b=0.)
    x0=torch.randn(1,4);x1=torch.randn(1,4);time=torch.tensor([0.5])
    core.loss(x0,x1,time).backward()
    active=(core.S.grad.abs().sum((1,2))>1e-10)
    assert active.sum()==1


def test_prism_no_wta_experts_receive_no_specialization_gradient():
    core=PrismFlowCore(4,hidden=8,latent=3,experts=3,variant="no_wta")
    core.loss(torch.randn(3,4),torch.randn(3,4),torch.rand(3)).backward()
    assert core.S.grad is None or torch.equal(core.S.grad,torch.zeros_like(core.S.grad))
    assert core.R.grad is None or torch.equal(core.R.grad,torch.zeros_like(core.R.grad))


def test_native_configs_entrypoints_and_paper_citations():
    import importlib,json,pathlib
    root=pathlib.Path(__file__).resolve().parents[1]
    paths=list((root/'configs/models').glob('fm_native*.json'))
    assert len(paths)==16
    for path in paths:
        config=json.loads(path.read_text(encoding='utf-8'))
        module,symbol=config['entrypoint'].split(':')
        constructor=getattr(importlib.import_module(module),symbol)
        parameters=dict(config['parameters']);parameters['device']='cpu'
        assert constructor(**parameters) is not None
        assert config['citation'].startswith('https://')
        assert config['paper_score_status']=='not_reproduced'
        assert config['design_choices']


@pytest.mark.parametrize("variant",["full","vanilla_fm","no_wta","no_balance","beta_zero","gamma_zero"])
def test_prism_variants_fit_and_sample(variant):
    x=np.random.default_rng(2).normal(size=(6,3,2)).astype(np.float32)
    model=PrismFlowModel(window=3,hidden=8,latent=3,experts=2,epochs=1,batch_size=3,
                         steps=3,variant=variant).fit(x)
    samples=model.sample(4)
    assert samples.shape==(4,3,2) and np.isfinite(samples).all()
    np.testing.assert_array_equal(samples,model.sample(4))
    if variant in {"vanilla_fm","gamma_zero"}:
        state=torch.randn(3,6);time=torch.rand(3)
        torch.testing.assert_close(model.model(time,state),model.model.global_field(time,state))


def test_invalid_graph_and_variants_rejected():
    with pytest.raises(ValueError):normalized_laplacian([[0,1],[0,0]])
    with pytest.raises(ValueError):GraphSpectralPath(-torch.eye(2))
    with pytest.raises(ValueError):GRASPDetector(variant="unknown")
