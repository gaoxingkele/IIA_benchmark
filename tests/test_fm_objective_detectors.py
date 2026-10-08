"""Analytic path/transport checks and CPU adapter behavior, not benchmarks."""
import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_objective_detectors import (
    CFMObjectiveDetector, OBJECTIVES, minibatch_plan, probability_path,
)


@pytest.mark.parametrize('objective',OBJECTIVES)
def test_conditional_velocity_matches_path_time_derivative(objective):
    x0=torch.tensor([[1.,2.],[3.,-1.]],dtype=torch.float64)
    x1=x0+2;epsilon=torch.tensor([[.1,-.2],[.5,.4]],dtype=torch.float64)
    t=torch.tensor([.3,.7],dtype=torch.float64);sigma=.4
    xt,ut,_=probability_path(x0,x1,t,epsilon,objective,sigma)
    dt=1e-5
    forward=probability_path(x0,x1,t+dt,epsilon,objective,sigma)[0]
    backward=probability_path(x0,x1,t-dt,epsilon,objective,sigma)[0]
    torch.testing.assert_close(ut,(forward-backward)/(2*dt),rtol=1e-6,atol=1e-6)
    if objective=='target':
        # Target-CFM ignores x0, using epsilon as its Gaussian source.
        torch.testing.assert_close(xt,t[:,None]*x1+(1-(1-sigma)*t[:,None])*epsilon)


def test_exact_ot_is_minimum_cost_and_sinkhorn_has_uniform_marginals():
    x0=torch.tensor([[0.],[2.],[5.]])
    x1=x0[[2,0,1]]
    plan=minibatch_plan(x0,x1)
    np.testing.assert_allclose(plan.sum(0),np.full(3,1/3))
    np.testing.assert_allclose(plan.sum(1),np.full(3,1/3))
    assert np.sum(plan*(torch.cdist(x0,x1).numpy()**2))==0
    soft=minibatch_plan(x0,x1,'sinkhorn',regularization=5.)
    np.testing.assert_allclose(soft.sum(0),np.full(3,1/3),atol=1e-6)
    np.testing.assert_allclose(soft.sum(1),np.full(3,1/3),atol=1e-6)
    assert np.all(soft>0)


@pytest.mark.parametrize('objective',OBJECTIVES)
def test_training_scoring_and_ode_use_frozen_local_randomness(objective):
    values=np.random.default_rng(17).normal(size=(8,4,2)).astype(np.float32)
    model=CFMObjectiveDetector(objective=objective,sigma=1.0 if objective in ('sb','sf2m') else .1,
                              window=4,hidden_size=8,epochs=1,batch_size=8,
                              monte_carlo_samples=1,score_times=(.25,.75),device='cpu')
    before=torch.random.get_rng_state().clone()
    model.fit(values)
    torch.testing.assert_close(torch.random.get_rng_state(),before)
    assert len(model.training_losses_)==1 and np.isfinite(model.training_losses_).all()
    np.testing.assert_array_equal(model.score(values),model.score(values))
    assert model.score(values).shape==(8,4)
    assert model.score(np.empty((0,4,2))).shape==(0,4)
    assert model.sample(3,steps=2).shape==(3,4,2)
    with pytest.raises(ValueError):model.score(np.ones((3,4,3)))


def test_bridge_and_input_errors_are_explicit():
    with pytest.raises(ValueError):CFMObjectiveDetector(objective='sb',sigma=0)
    with pytest.raises(ValueError):CFMObjectiveDetector(score_times=(0.,))
    model=CFMObjectiveDetector(window=4,device='cpu')
    with pytest.raises(RuntimeError):model.score(np.ones((2,4,2)))
    with pytest.raises(ValueError):model.fit(np.full((2,4,2),np.nan))
    with pytest.raises(RuntimeError,match='converge'):
        minibatch_plan(torch.tensor([[0.],[1.],[3.]]),torch.tensor([[0.],[.2],[2.]]),'sinkhorn',2.,max_iterations=1)


def test_sf2m_flow_only_ablation_removes_score_head():
    values=np.zeros((4,3,1),dtype=np.float32)
    model=CFMObjectiveDetector(objective='sf2m',sigma=1.,score_weight=0.,window=3,
                              hidden_size=4,batch_size=4,epochs=1,device='cpu').fit(values)
    assert not model._score_head
    assert model._model[-1].out_features==3
