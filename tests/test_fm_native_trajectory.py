import numpy as np
import pytest
from iia_benchmark.models.fm_native_trajectory import (
    kalman_rts,brownian_bridge_targets,gp_joint_posterior,gp_targets,
    CFMTSModel,
)


def test_kalman_smoother_reduces_noise_on_constant_state():
    y=3+np.random.default_rng(5).normal(scale=0.2,size=(100,1))
    m,p=kalman_rts(y,process_noise=0.0001,observation_noise=0.04)
    assert np.mean((m-3)**2)<np.mean((y-3)**2)
    assert (p>0).all()


def test_bb_deterministic_mean_derivative():
    t=np.linspace(0,1,6);y=(2*t+3)[:,None];q=np.array([0.15,0.45,0.75])
    m,p=kalman_rts(y)
    x,u=brownian_bridge_targets(t,y,q,np.zeros((3,1)),mode="gaussian_consistent")
    i=np.searchsorted(t,q)-1
    expected=(m[i+1]-m[i])/(t[i+1]-t[i])[:,None]
    np.testing.assert_allclose(u,expected)
    # Literal and corrected covariance transport differ; preserve that boundary.
    _,literal=brownian_bridge_targets(t,y,q,np.ones((3,1)),mode="paper_literal")
    _,corrected=brownian_bridge_targets(t,y,q,np.ones((3,1)),mode="gaussian_consistent")
    assert not np.allclose(literal,corrected)


def test_gp_derivative_mean_matches_finite_difference():
    t=np.linspace(0,2,10);y=np.sin(t);q=np.linspace(0.1,1.9,7);h=1e-5
    m,dm,cov,dcov=gp_joint_posterior(t,y,q,lengthscale=0.8)
    plus=gp_joint_posterior(t,y,q+h,lengthscale=0.8)[0]
    minus=gp_joint_posterior(t,y,q-h,lengthscale=0.8)[0]
    np.testing.assert_allclose(dm,(plus-minus)/(2*h),rtol=1e-6,atol=1e-6)
    assert np.linalg.eigvalsh(cov).min()>-1e-10
    x,u=gp_targets(t,y,q,np.zeros((7,1)),lengthscale=0.8)
    np.testing.assert_allclose(x,m);np.testing.assert_allclose(u,dm)


@pytest.mark.parametrize("method",["bb","gp"])
def test_cfmts_train_and_simulate(method):
    t=np.linspace(0,1,12);y=np.sin(t)[:,None]
    model=CFMTSModel(method=method,hidden=8,epochs=2,batch_size=8,samples=2,
                     grid_points=10,gp_max_iterations=3).fit([(t,y)])
    prediction=model.simulate([0.],np.linspace(0,1,8))
    assert prediction.shape==(8,1) and np.isfinite(prediction).all()
    assert len(model.loss_history_)==2
