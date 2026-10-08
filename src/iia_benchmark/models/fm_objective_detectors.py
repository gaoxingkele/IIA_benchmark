"""Trainable FM objective adapters with explicitly local TSAD scoring.

Path/velocity targets transcribe TorchCFM commit
7c653857de4c25b979a740ef8b6a5d5ddb09b6c1 conditional_flow_matching.py.
Exact equal-mass OT uses SciPy assignment; entropic OT uses log-Sinkhorn.
These replace POT numerics, not the mathematical coupling. Independent pair
velocity residuals are a LOCAL anomaly-score adapter, not a paper likelihood
or a reproduction of original image/single-cell experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

OBJECTIVES=('independent','ot','target','sb','vp','sf2m','rectified')


def probability_path(x0, x1, t, epsilon, objective='independent', sigma=0.0):
    """Return xt, conditional velocity and path std, for fixed endpoints/noise.

    Endpoint coupling is separate so formulas can be checked without a solver.
    t is strictly inside (0,1) for Brownian bridge targets.
    """
    import torch
    if objective not in OBJECTIVES:raise ValueError('Unknown FM objective')
    if sigma < 0 or not math.isfinite(sigma):raise ValueError('Invalid path sigma')
    if x0.shape!=x1.shape or epsilon.shape!=x0.shape or t.shape!=(len(x0),):
        raise ValueError('Path tensor shapes differ')
    if not bool(torch.isfinite(t).all()) or bool(((t<0)|(t>1)).any()):
        raise ValueError('Path time must be in [0,1]')
    time=t.reshape((-1,)+(1,)*(x0.ndim-1))
    if objective=='target':
        std=1-(1-sigma)*time
        if bool((std<=0).any()):raise ValueError('Singular target path endpoint')
        xt=time*x1+std*epsilon
        velocity=(x1-(1-sigma)*xt)/std
    elif objective in ('sb','sf2m'):
        if sigma<=0 or bool(((time<=0)|(time>=1)).any()):
            raise ValueError('Bridge needs positive sigma and interior times')
        mean=(1-time)*x0+time*x1
        std=sigma*torch.sqrt(time*(1-time))
        xt=mean+std*epsilon
        velocity=x1-x0+(1-2*time)/(2*time*(1-time)+1e-8)*(xt-mean)
    elif objective=='vp':
        angle=time*math.pi/2
        std=torch.full_like(time,sigma)
        xt=torch.cos(angle)*x0+torch.sin(angle)*x1+std*epsilon
        velocity=math.pi/2*(torch.cos(angle)*x1-torch.sin(angle)*x0)
    else:
        std=torch.full_like(time,0.0 if objective=='rectified' else sigma)
        xt=(1-time)*x0+time*x1+std*epsilon
        velocity=x1-x0
    return xt,velocity,std


def minibatch_plan(x0, x1, method='exact', regularization=2.0,
                   tolerance=1e-6, max_iterations=2000):
    """Uniform equal-size squared-distance OT plan; fail on bad marginals.

    No uniform fallback: a failed transport computation must be visible.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.special import logsumexp
    import torch
    if len(x0)!=len(x1) or len(x0)==0:raise ValueError('OT needs equal nonempty batches')
    if max_iterations<=0 or tolerance<=0:raise ValueError('Invalid solver stopping rule')
    cost=(torch.cdist(x0.flatten(1).double(),x1.flatten(1).double())**2).detach().cpu().numpy()
    if not np.isfinite(cost).all():raise ValueError('Nonfinite OT cost')
    n=len(cost)
    if method=='exact':
        rows,cols=linear_sum_assignment(cost)
        plan=np.zeros_like(cost);plan[rows,cols]=1/n
        return plan
    if method!='sinkhorn':raise ValueError('Unknown coupling solver')
    if regularization<=0 or not np.isfinite(regularization):raise ValueError('Invalid entropy regularization')
    kernel=-cost/regularization
    log_mass=-math.log(n);log_v=np.zeros(n)
    for iteration in range(max_iterations):
        log_u=log_mass-logsumexp(kernel+log_v[None,:],axis=1)
        log_v=log_mass-logsumexp(kernel+log_u[:,None],axis=0)
        if iteration%10==0 or iteration==max_iterations-1:
            plan=np.exp(kernel+log_u[:,None]+log_v[None,:])
            error=max(np.abs(plan.sum(0)-1/n).max(),np.abs(plan.sum(1)-1/n).max())
            if error<=tolerance:return plan
    raise RuntimeError(f'Sinkhorn did not converge: marginal error={error:.3g}')


def coupled_endpoints(x0, x1, objective, sigma, generator):
    import torch
    if objective not in ('ot','sb','sf2m'):return x0,x1
    plan=minibatch_plan(x0,x1,'exact' if objective=='ot' else 'sinkhorn',2*sigma**2)
    weights=torch.as_tensor(plan.flatten(),dtype=torch.float64,device=x0.device)
    pairs=torch.multinomial(weights,len(x0),replacement=True,generator=generator)
    return x0[pairs//len(x1)],x1[pairs%len(x1)]


@dataclass
class CFMObjectiveDetector:
    """FM training + Monte Carlo conditional-velocity discrepancy per window.

    Inputs are already train-normalized windows [N,L,C]. No labels, dataset
    paths, test-derived graph/threshold or fitted test statistics are consumed.
    A shared MLP backbone freezes capacity across seven objective adapters.
    ``sample`` supports Euler/Heun forward ODE; it is not a likelihood routine.
    """
    objective: str='independent'
    sigma: float=0.0
    window: int=100
    hidden_size: int=128
    epochs: int=20
    batch_size: int=64
    learning_rate: float=1e-3
    score_times: tuple=(0.1,0.3,0.5,0.7,0.9)
    monte_carlo_samples: int=4
    score_weight: float=1.0
    grad_clip: float=1.0
    seed: int=1103
    device: str='auto'
    kind: str=field(default='window',init=False)
    name: str=field(default='cfm_objective_adapter',init=False)
    _model: Any=field(default=None,init=False,repr=False)
    _shape: Any=field(default=None,init=False,repr=False)
    training_losses_: list=field(default_factory=list,init=False)

    def __post_init__(self):
        if self.objective not in OBJECTIVES:raise ValueError('Unknown FM objective')
        if self.sigma<0 or not np.isfinite(self.sigma):raise ValueError('Invalid sigma')
        if self.objective in ('sb','sf2m') and self.sigma<=0:raise ValueError('Bridge sigma must be positive')
        if min(self.window,self.hidden_size,self.epochs,self.batch_size,self.monte_carlo_samples)<=0:
            raise ValueError('Positive model and training sizes required')
        if self.learning_rate<=0 or self.score_weight<0:raise ValueError('Invalid loss or learning rate')
        if not self.score_times or any(not 0<float(t)<1 for t in self.score_times):
            raise ValueError('Scoring times must be strictly inside (0,1)')

    def _windows(self, values):
        array=np.asarray(values,dtype=np.float32)
        if array.ndim!=3 or array.shape[1]!=self.window or not array.shape[2]:
            raise ValueError('Expected windows [N,configured window,features]')
        if not np.isfinite(array).all():raise ValueError('Nonfinite input windows')
        if self._shape is not None and tuple(array.shape[1:])!=self._shape:
            raise ValueError('Feature or window shape differs from training')
        return array

    def fit(self, windows):
        import torch
        from torch import nn
        values=self._windows(windows)
        if not len(values):raise ValueError('No training windows')
        self._shape=tuple(values.shape[1:]);dimensions=int(np.prod(self._shape))
        self._device=torch.device('cuda' if self.device=='auto' and torch.cuda.is_available() else 'cpu' if self.device=='auto' else self.device)
        self._score_head=self.objective=='sf2m' and self.score_weight>0
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self._model=nn.Sequential(nn.Linear(dimensions+1,self.hidden_size),nn.SiLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size),nn.SiLU(),
                                      nn.Linear(self.hidden_size,dimensions*(2 if self._score_head else 1)))
        self._model.to(self._device)
        generator=torch.Generator(device=self._device).manual_seed(self.seed)
        # Keep the full training array on CPU; only a minibatch uses the GPU.
        tensor=torch.from_numpy(np.ascontiguousarray(values.reshape(len(values),-1)))
        optimizer=torch.optim.Adam(self._model.parameters(),lr=self.learning_rate)
        self.training_losses_=[]
        for _ in range(self.epochs):
            order=torch.randperm(len(values),generator=generator,device=self._device).cpu()
            total=0.0
            for start in range(0,len(values),self.batch_size):
                x1=tensor[order[start:start+self.batch_size]].to(self._device)
                x0=torch.randn(x1.shape,generator=generator,device=self._device)
                x0,x1=coupled_endpoints(x0,x1,self.objective,self.sigma,generator)
                t=torch.rand(len(x1),generator=generator,device=self._device).clamp(1e-4,1-1e-4)
                noise=torch.randn(x1.shape,generator=generator,device=self._device)
                xt,ut,std=probability_path(x0,x1,t,noise,self.objective,self.sigma)
                output=self._model(torch.cat([xt,t[:,None]],dim=1))
                loss=(output[:,:dimensions]-ut).square().mean()
                if self._score_head:
                    # TorchCFM SF2M tutorial: ||lambda(t)*s_theta + epsilon||².
                    weight=2*std/(self.sigma**2+1e-8)
                    loss=loss+self.score_weight*(weight*output[:,dimensions:]+noise).square().mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM training loss')
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(),self.grad_clip)
                optimizer.step();total+=float(loss.detach())*len(x1)
            self.training_losses_.append(total/len(values))
        self._model.eval()
        return self

    def score(self, windows):
        import torch
        if self._model is None:raise RuntimeError('fit must be called first')
        values=self._windows(windows)
        if not len(values):return np.empty((0,self.window),dtype=np.float32)
        dimensions=int(np.prod(self._shape));outputs=[]
        # Freeze this batching and RNG convention in the model configuration.
        generator=torch.Generator(device=self._device).manual_seed(self.seed+1)
        with torch.no_grad():
            for start in range(0,len(values),self.batch_size):
                x1=torch.as_tensor(values[start:start+self.batch_size].reshape(-1,dimensions),device=self._device)
                residual=torch.zeros_like(x1)
                for _ in range(self.monte_carlo_samples):
                    for time in self.score_times:
                        x0=torch.randn(x1.shape,generator=generator,device=self._device)
                        epsilon=torch.randn(x1.shape,generator=generator,device=self._device)
                        t=torch.full((len(x1),),float(time),device=self._device)
                        xt,ut,_=probability_path(x0,x1,t,epsilon,self.objective,self.sigma)
                        prediction=self._model(torch.cat([xt,t[:,None]],dim=1))[:,:dimensions]
                        residual+=(prediction-ut).square()
                residual/=self.monte_carlo_samples*len(self.score_times)
                outputs.append(residual.reshape((-1,)+self._shape).mean(-1).cpu().numpy())
        result=np.concatenate(outputs)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite FM score')
        return result

    def sample(self, count, steps=50, solver='heun'):
        import torch
        if self._model is None:raise RuntimeError('fit must be called first')
        if count<=0 or steps<=0 or solver not in ('euler','heun'):raise ValueError('Invalid ODE sampler settings')
        dimensions=int(np.prod(self._shape))
        generator=torch.Generator(device=self._device).manual_seed(self.seed+2)
        with torch.no_grad():
            x=torch.randn((count,dimensions),generator=generator,device=self._device)
            for step in range(steps):
                t=torch.full((count,1),step/steps,device=self._device)
                velocity=self._model(torch.cat([x,t],dim=1))[:,:dimensions]
                candidate=x+velocity/steps
                if solver=='heun':
                    next_t=torch.full_like(t,(step+1)/steps)
                    next_velocity=self._model(torch.cat([candidate,next_t],dim=1))[:,:dimensions]
                    x=x+(velocity+next_velocity)/(2*steps)
                else:x=candidate
        return x.reshape((count,)+self._shape).cpu().numpy()
