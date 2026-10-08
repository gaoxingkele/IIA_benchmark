"""CFM-TS reconstruction (OpenReview Hqn4Aj7xrQ, algorithms 1--2).

The supplied paper has ambiguous/inconsistent BB covariance, RTS, and GP mean
formulas. Literal targets and corrected Gaussian targets are separately named;
neither is certified identical to unreleased author code.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn import functional as F
from .fm_native import TimeMLP


def kalman_rts(observations,process_noise=0.005,observation_noise=0.001):
    """Zero-drift diagonal random-walk filter/standard RTS (Appendix B repair).

    Q/R are variances. The initial prior m=y0,P=1 is an explicit local choice.
    Correct RTS uses predicted covariance inverse and G on both sides; the
    printed appendix equations omit/mix these terms.
    """
    y=np.asarray(observations,dtype=float)
    if y.ndim==1: y=y[:,None]
    if len(y)<2 or not np.isfinite(y).all() or process_noise<0 or observation_noise<=0:
        raise ValueError("Need >=2 finite observations, Q>=0, R>0")
    means=np.empty_like(y); variances=np.empty_like(y)
    predicted=np.empty_like(y); predvar=np.empty_like(y)
    m=y[0].copy();p=np.ones(y.shape[1])
    for i in range(len(y)):
        if i: p=p+process_noise
        predicted[i],predvar[i]=m,p
        gain=p/(p+observation_noise)
        m=m+gain*(y[i]-m);p=(1-gain)*p
        means[i],variances[i]=m,p
    smooth_m,smooth_p=means.copy(),variances.copy()
    for i in range(len(y)-2,-1,-1):
        gain=variances[i]/predvar[i+1]
        smooth_m[i]=means[i]+gain*(smooth_m[i+1]-predicted[i+1])
        smooth_p[i]=variances[i]+gain**2*(smooth_p[i+1]-predvar[i+1])
    return smooth_m,np.maximum(smooth_p,1e-12)


def brownian_bridge_targets(times,observations,query_times,noise,mode="paper_literal",
                            process_noise=0.005,observation_noise=0.001):
    """Algorithm-1 marginal interpolation, with both target interpretations.

    paper_literal: Eq.8 dP/dt*(x-m0)/P0 + dm/dt.
    gaussian_consistent: dm/dt + 0.5*dP/dt/P(t)*(x-m(t)).
    The latter transports the printed Gaussian Eq.7 but differs from Eq.8.
    Independent marginal sampling is not a Brownian trajectory joint sampler.
    """
    t=np.asarray(times,dtype=float);q=np.asarray(query_times,dtype=float)
    if t.ndim!=1 or len(t)<2 or len(t)!=len(observations) or not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(np.diff(t)<=0) or np.any((q<t[0])|(q>t[-1])):
        raise ValueError("Need ordered times and in-range queries")
    if mode not in {"paper_literal","gaussian_consistent"}: raise ValueError("Unknown BB mode")
    m,p=kalman_rts(observations,process_noise,observation_noise)
    idx=np.searchsorted(t,q,side="right").clip(1,len(t)-1)-1
    dt=(t[idx+1]-t[idx])[:,None]; lam=((q-t[idx])[:,None]/dt)
    dm=m[idx+1]-m[idx];dp=p[idx+1]-p[idx]
    mean=m[idx]+lam*dm;variance=p[idx]+lam*dp
    eps=np.asarray(noise,dtype=float)
    if eps.shape!=mean.shape: raise ValueError("Noise must have shape [queries, state dimensions]")
    x=mean+np.sqrt(variance)*eps
    if mode=="paper_literal": velocity=dm/dt+(dp/dt)*(x-m[idx])/p[idx]
    else: velocity=dm/dt+0.5*(dp/dt)*(x-mean)/variance
    return x,velocity


def gp_joint_posterior(times,observations,query_times,lengthscale=1.,variance=1.,noise_variance=0.001):
    """RBF joint state/derivative posterior with observation noise and means."""
    t=np.asarray(times,dtype=float);q=np.asarray(query_times,dtype=float)
    y=np.asarray(observations,dtype=float)
    if y.ndim==1: y=y[:,None]
    if min(lengthscale,variance,noise_variance)<=0 or len(t)!=len(y) or len(t)<2 or not np.isfinite(y).all() or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("Need positive GP parameters and finite matched observations")
    def kernel(a,b):
        diff=a[:,None]-b[None,:]
        return variance*np.exp(-0.5*diff**2/lengthscale**2),diff
    ktt,_=kernel(t,t); ktt+=noise_variance*np.eye(len(t))
    kqt,dqt=kernel(q,t);dqt_kernel=-dqt/lengthscale**2*kqt
    kqq,dqq=kernel(q,q)
    alpha=np.linalg.solve(ktt,y)
    mean=kqt@alpha; derivative_mean=dqt_kernel@alpha
    cov=kqq-kqt@np.linalg.solve(ktt,kqt.T)
    # Cov(d x(q)/dq, x(q')) = derivative in first kernel argument.
    dcov=-dqq/lengthscale**2*kqq-dqt_kernel@np.linalg.solve(ktt,kqt.T)
    return mean,derivative_mean,(cov+cov.T)/2,dcov


def fit_rbf_hyperparameters(times,observations,noise_variance=0.001,max_iterations=400):
    """Per-trajectory MLE of variance/lengthscale (Appendix C/D)."""
    t=np.asarray(times,dtype=float);y=np.asarray(observations,dtype=float)
    if y.ndim==1: y=y[:,None]
    diff=t[:,None]-t[None,:]
    def nll(parameters):
        length,var=np.exp(parameters)
        cov=var*np.exp(-0.5*diff**2/length**2)+noise_variance*np.eye(len(t))
        try:
            chol=np.linalg.cholesky(cov)
            solve=np.linalg.solve(chol,y)
            return 0.5*np.square(solve).sum()+y.shape[1]*np.log(np.diag(chol)).sum()
        except np.linalg.LinAlgError: return 1e100
    result=minimize(nll,np.log([max(np.ptp(t)/4,1e-3),max(float(np.var(y)),1e-3)]),
                    method="L-BFGS-B",bounds=[(-8,8),(-12,12)],options={"maxiter":max_iterations})
    return {"lengthscale":float(np.exp(result.x[0])),"variance":float(np.exp(result.x[1])),
            "optimizer_success":bool(result.success),"objective":float(result.fun)}


def gp_targets(times,observations,query_times,noise,mode="gaussian_conditional",**parameters):
    """Algorithm-2 joint GP target, literal Eq.10 also available.

    gaussian_conditional: m_dot+C_dotx C_xx^-1(x-m).
    paper_literal: C_dotx C_xx^-1 x, the zero-mean printed expression.
    Independent output dimensions share one kernel; entire query vector sampled
    jointly, so covariance across times is retained (unlike marginal BB calls).
    """
    if mode not in {"gaussian_conditional","paper_literal"}: raise ValueError("Unknown GP mode")
    mean,dmean,cov,dcov=gp_joint_posterior(times,observations,query_times,**parameters)
    eps=np.asarray(noise,dtype=float)
    if eps.shape!=mean.shape: raise ValueError("Noise dimensions must match posterior mean")
    stable=cov+1e-9*np.eye(len(cov))
    x=mean+np.linalg.cholesky(stable)@eps
    if mode=="paper_literal": velocity=dcov@np.linalg.solve(stable,x)
    else: velocity=dmean+dcov@np.linalg.solve(stable,x-mean)
    return x,velocity


class CFMTSModel:
    """Train a callable trajectory vector field on BB or GP targets.

    fit takes [(times, observations), ...], simulate takes initial state and
    physical times. Data splits and dataset equation choices remain in configs.
    The flat TimeMLP differs from ambiguous 64/258-dimension paper architecture;
    this is an executable reconstruction, not a paper-score claim.
    """
    def __init__(self,method="bb",target_mode=None,hidden=258,epochs=400,batch_size=256,
                 learning_rate=0.01,samples=10,grid_points=50,seed=1103,device="cpu",
                 gp_max_iterations=400):
        if method not in {"bb","gp"}: raise ValueError("Choose bb or gp")
        if min(hidden,epochs,batch_size,samples,grid_points,gp_max_iterations)<1:
            raise ValueError("Positive dimensions/budgets required")
        self.method=method
        self.target_mode=target_mode or ("paper_literal" if method=="bb" else "gaussian_conditional")
        self.hidden,self.epochs,self.batch_size=hidden,epochs,batch_size
        self.learning_rate,self.samples,self.grid_points=learning_rate,samples,grid_points
        self.seed,self.device,self.gp_max_iterations=seed,torch.device(device),gp_max_iterations

    def fit(self,trajectories):
        rng=np.random.default_rng(self.seed);states=[];velocities=[];times=[]
        self.preprocessing_=[]
        for observed_times,y in trajectories:
            y=np.asarray(y,dtype=float)
            if y.ndim==1: y=y[:,None]
            q=np.linspace(observed_times[0],observed_times[-1],self.grid_points)
            parameters=fit_rbf_hyperparameters(observed_times,y,max_iterations=self.gp_max_iterations) if self.method=="gp" else {}
            self.preprocessing_.append(parameters)
            for _ in range(self.samples):
                eps=rng.normal(size=(len(q),y.shape[1]))
                if self.method=="bb":x,u=brownian_bridge_targets(observed_times,y,q,eps,mode=self.target_mode)
                else:x,u=gp_targets(observed_times,y,q,eps,mode=self.target_mode,lengthscale=parameters["lengthscale"],variance=parameters["variance"])
                states.append(x);velocities.append(u);times.append(q)
        if not states: raise ValueError("No trajectories")
        x=torch.tensor(np.concatenate(states),dtype=torch.float32)
        u=torch.tensor(np.concatenate(velocities),dtype=torch.float32)
        t=torch.tensor(np.concatenate(times),dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model=TimeMLP(x.shape[1],self.hidden).to(self.device)
            optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
            self.loss_history_=[]
            for _ in range(self.epochs):
                total=0.
                for ids in torch.randperm(len(x)).split(self.batch_size):
                    loss=F.mse_loss(self.model(t[ids].to(self.device),x[ids].to(self.device)),u[ids].to(self.device))
                    optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
                self.loss_history_.append(total/len(x))
        self.model.eval()
        return self

    def vector_field(self,time,state):
        if not hasattr(self,"model"): raise RuntimeError("Call fit first")
        x=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        t=torch.as_tensor(time,dtype=torch.float32,device=self.device).expand(len(x))
        with torch.no_grad(): return self.model(t,x).cpu().numpy()

    def simulate(self,initial_state,times):
        """RK4 diagnostic integration; dopri5 paper evaluation remains pending."""
        t=np.asarray(times,dtype=float)
        if np.any(np.diff(t)<=0): raise ValueError("Times must increase")
        x=np.asarray(initial_state,dtype=float).reshape(-1);out=[x.copy()]
        for i in range(len(t)-1):
            dt=t[i+1]-t[i]
            f=lambda time,state:self.vector_field(time,state[None,:])[0]
            k1=f(t[i],x);k2=f(t[i]+dt/2,x+dt*k1/2);k3=f(t[i]+dt/2,x+dt*k2/2);k4=f(t[i+1],x+dt*k3)
            x=x+dt*(k1+2*k2+2*k3+k4)/6;out.append(x.copy())
        return np.stack(out)
