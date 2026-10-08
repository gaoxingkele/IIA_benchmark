"""Paper-derived native FM implementations, not certified author reproductions.

GRASP: arXiv:2609.36765 equations 10--19, 27--28, Appendix C.
DFM: arXiv:2410.09246 equation 12 and continuous change of variables.
PrismFlow: arXiv:2605.28867 equations 5--14.
All dataset preparation and evaluation policy belongs to configs/the harness.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _windows(values, window):
    x = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    if x.ndim != 3 or x.shape[1] != window or not torch.isfinite(x).all():
        raise ValueError("Expected finite windows [N, window, C]")
    if len(x) == 0:
        raise ValueError("Empty windows")
    return x


def normalized_laplacian(adjacency):
    a = torch.as_tensor(adjacency)
    if not a.is_floating_point():
        a = a.float()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not torch.isfinite(a).all():
        raise ValueError("Adjacency must be a finite square matrix")
    if torch.any(a < 0) or not torch.allclose(a, a.T):
        raise ValueError("Adjacency must be nonnegative and symmetric")
    a = a.clone()
    a.fill_diagonal_(0)
    d = a.sum(-1)
    inv = torch.where(d > 0, d.clamp_min(1e-30).rsqrt(), torch.zeros_like(d))
    # Isolated nodes have a zero diagonal and thus a zero-frequency limit.
    return torch.diag((d > 0).to(a.dtype)) - inv[:, None] * a * inv[None, :]


class GraphSpectralPath(nn.Module):
    """Exact GRASP path in [B, C, L], including the zero-frequency limit."""
    def __init__(self, laplacian, tau=2.0):
        super().__init__()
        lap = torch.as_tensor(laplacian)
        if not lap.is_floating_point():
            lap = lap.float()
        if not math.isfinite(tau) or tau < 0 or lap.ndim != 2 or lap.shape[0] != lap.shape[1] or not torch.isfinite(lap).all() or not torch.allclose(lap, lap.T):
            raise ValueError("Need symmetric finite PSD Laplacian and tau >= 0")
        vals, basis = torch.linalg.eigh(lap)
        if vals.min() < -1e-6:
            raise ValueError("Laplacian must be positive semidefinite")
        self.register_buffer("basis", basis)
        self.register_buffer("omega", (tau * vals.clamp_min(0)).sqrt())

    def coefficients(self, time):
        t = torch.as_tensor(time, dtype=self.omega.dtype, device=self.omega.device).reshape(-1, 1, 1)
        if torch.any((t < 0) | (t > 1)):
            raise ValueError("Flow times must lie in [0,1]")
        w = self.omega[None, :, None]
        # Safe denominators prevent NaNs in the non-selected torch.where branch.
        positive = w > 1e-7
        safe = torch.where(positive, w, torch.ones_like(w))
        denominator = safe.sinh()
        a = torch.where(positive, (safe * (1-t)).sinh()/denominator, 1-t)
        b = torch.where(positive, (safe * t).sinh()/denominator, t)
        da = torch.where(positive, -safe*(safe*(1-t)).cosh()/denominator, -torch.ones_like(t))
        db = torch.where(positive, safe*(safe*t).cosh()/denominator, torch.ones_like(t))
        return a, b, da, db

    def spectral(self, x):
        return torch.einsum("ck,bcl->bkl", self.basis, x)

    def node(self, x):
        return torch.einsum("ck,bkl->bcl", self.basis, x)

    def forward(self, source, target, time):
        a, b, da, db = self.coefficients(time)
        x0, x1 = self.spectral(source), self.spectral(target)
        return self.node(a*x0+b*x1), self.node(da*x0+db*x1)

    def weights(self, time):
        t = torch.as_tensor(time, device=self.omega.device, dtype=self.omega.dtype).reshape(-1,1,1)
        w = self.omega[None,:,None]
        safe = torch.where(w > 1e-7, w, torch.ones_like(w))
        return torch.where(w > 1e-7, ((safe*t).sinh()/safe).square(), t.square())


class _MixBlock(nn.Module):
    def __init__(self, channels, length, hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm((channels,length))
        self.norm2 = nn.LayerNorm((channels,length))
        self.temporal = nn.Sequential(nn.Linear(length,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,length))
        self.variable = nn.Sequential(nn.Linear(channels,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,channels))

    def forward(self, x):
        x = x+self.temporal(self.norm1(x))
        return x+self.variable(self.norm2(x).transpose(-1,-2)).transpose(-1,-2)


class MixerVelocity(nn.Module):
    """Appendix-C-style TSMixer; unspecified projection axes are explicit choices."""
    def __init__(self, channels, window, hidden=128, blocks=2, dropout=0.1):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(-math.log(10000)*torch.arange(32)/32))
        self.input = nn.Linear(window+64,window)
        self.blocks = nn.Sequential(*[_MixBlock(channels,window,hidden,dropout) for _ in range(blocks)])
        self.output = nn.Linear(window,window)

    def forward(self, time, x):
        t = time.reshape(-1,1)*self.frequencies[None,:]
        e = torch.cat((t.sin(),t.cos()),-1)[:,None,:].expand(-1,x.shape[1],-1)
        return self.output(self.blocks(self.input(torch.cat((x,e),-1))))


class GRASPDetector:
    """Trainable GRASP reconstruction; no test labels or test calibration used.

    Graph estimation is unresolved in the paper: callers can provide adjacency.
    Otherwise abs-Pearson thresholding is an explicit local design choice.
    GRASP-ER/l in paper are applied on top of GRASP-mean, not weighted GRASP.
    """
    def __init__(self, window=50, hidden=128, blocks=2, dropout=0.1, tau=2.0,
                 graph_threshold=0.5, variant="grasp", epochs=1500, batch_size=256,
                 learning_rate=0.001, source_samples=5, flow_evaluations=10,
                 seed=1103, device="auto"):
        if variant not in {"grasp","mean","er","lin"}:
            raise ValueError("Unknown GRASP variant")
        if min(window,epochs,batch_size,source_samples,flow_evaluations) < 1:
            raise ValueError("Budgets must be positive")
        self.window,self.hidden,self.blocks,self.dropout=window,hidden,blocks,dropout
        self.tau,self.graph_threshold,self.variant=tau,graph_threshold,variant
        self.epochs,self.batch_size,self.learning_rate=epochs,batch_size,learning_rate
        self.source_samples,self.flow_evaluations=source_samples,flow_evaluations
        self.seed=seed
        self.device=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device)

    def fit(self, windows, adjacency=None):
        x=_windows(windows,self.window).transpose(1,2)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=="cuda" else []):
            torch.manual_seed(self.seed)
            if adjacency is None:
                flat=x.transpose(1,2).reshape(-1,x.shape[1]).double()
                corr=torch.corrcoef(flat.T).nan_to_num(0).abs()
                a=(corr>=self.graph_threshold).float()
                self.graph_source="train_only_abs_pearson_local_choice"
            else:
                a=torch.as_tensor(adjacency,dtype=torch.float32).clone()
                self.graph_source="caller_supplied"
            if a.shape!=(x.shape[1],x.shape[1]):
                raise ValueError("Adjacency dimensions differ from channels")
            a.fill_diagonal_(0)
            if self.variant=="er":
                # Match exact edge count, sample uniformly without replacement.
                i,j=torch.triu_indices(len(a),len(a),1)
                count=int((a[i,j]>0).sum())
                order=torch.randperm(len(i))[:count]
                a=torch.zeros_like(a)
                a[i[order],j[order]]=1; a[j[order],i[order]]=1
            self.adjacency_=a.numpy().copy()
            self.path=GraphSpectralPath(normalized_laplacian(a),0 if self.variant=="lin" else self.tau).to(self.device)
            self.model=MixerVelocity(x.shape[1],self.window,self.hidden,self.blocks,self.dropout).to(self.device)
            optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
            self.loss_history_=[]
            self.model.train()
            for _ in range(self.epochs):
                total=0.
                for ids in torch.randperm(len(x)).split(self.batch_size):
                    target=x[ids].to(self.device)
                    source=torch.randn_like(target); t=torch.rand(len(target),device=self.device)
                    state,velocity=self.path(source,target,t)
                    loss=F.mse_loss(self.model(t,state),velocity)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total+=float(loss.detach())*len(ids)
                self.loss_history_.append(total/len(x))
        self.model.eval()
        return self

    def score(self, windows):
        if not hasattr(self,"model"):
            raise RuntimeError("Call fit before score")
        x=_windows(windows,self.window).transpose(1,2)
        # Common random sources across windows make score independent of chunking.
        rng=torch.Generator(device=self.device).manual_seed(self.seed+1)
        sources=torch.randn((self.source_samples,1,x.shape[1],self.window),generator=rng,device=self.device)
        times=torch.linspace(0,1,self.flow_evaluations+2,device=self.device)[1:-1]
        scores=[]
        with torch.no_grad():
            for target in x.split(self.batch_size):
                target=target.to(self.device); result=torch.zeros(len(target),device=self.device)
                for source in sources:
                    for t in times:
                        ts=t.expand(len(target)); state,velocity=self.path(source.expand_as(target),target,ts)
                        residual=self.path.spectral(self.model(ts,state)-velocity).square()
                        weight=self.path.weights(ts) if self.variant=="grasp" else 1.
                        result+=(residual*weight).sum((1,2))
                scores.append((result/self.source_samples).cpu().numpy())
        return np.concatenate(scores)


class TimeMLP(nn.Module):
    def __init__(self, dimensions, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dimensions+1,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,dimensions))
    def forward(self,time,x):
        return self.net(torch.cat((x,time.reshape(-1,1)),dim=-1))


def dual_cosine_loss(forward_velocity, reverse_velocity):
    """Literal DFM Eq.12. Alignment alone does NOT guarantee inverse maps."""
    return (1-F.cosine_similarity(forward_velocity,reverse_velocity,dim=-1)).mean()


def cnf_log_density(field, observations, steps=4, trace="exact", generator=None):
    """Euler backward solve and log-det accumulation with the correct sign.

    Returns log p1(x1)=log p0(x0)-integral_0^1 div(v)dt.
    Unit tests use affine fields with an analytic density. Exact trace is a local
    diagnostic; paper uses Hutchinson and also Dopri5 (not implemented here).
    """
    if steps<1 or trace not in {"exact","hutchinson"}:
        raise ValueError("Need steps>=1 and exact/hutchinson trace")
    x=observations.detach().clone(); correction=torch.zeros(len(x),device=x.device,dtype=x.dtype)
    eps=None
    if trace=="hutchinson":
        eps=torch.randint(0,2,x.shape,device=x.device,generator=generator).to(x.dtype)*2-1
    with torch.enable_grad():
        for i in range(steps):
            x=x.detach().requires_grad_(True)
            t=torch.full((len(x),),1-i/steps,device=x.device,dtype=x.dtype)
            v=field(t,x)
            if not v.requires_grad:
                div=torch.zeros_like(correction)
            elif trace=="exact":
                div=torch.zeros_like(correction)
                for j in range(x.shape[1]):
                    grad=torch.autograd.grad(v[:,j].sum(),x,retain_graph=True,allow_unused=True)[0]
                    if grad is not None: div=div+grad[:,j]
            else:
                grad=torch.autograd.grad((v*eps).sum(),x,allow_unused=True)[0]
                div=torch.zeros_like(correction) if grad is None else (grad*eps).sum(-1)
            correction=correction-div.detach()/steps
            x=(x-v/steps).detach()
    log_prior=-0.5*(x.square()+math.log(2*math.pi)).sum(-1)
    return log_prior+correction


class DFMDetector:
    """Callable literal objective reconstruction, with documented limitations.

    Paper specifies U-Net and learnable Gaussian; this core uses TimeMLP and a
    fixed standard prior. Eq.12 admits degenerate aligned fields, so losses and
    fit/score availability must never be interpreted as scientific replication.
    """
    def __init__(self,window=8,hidden=128,epochs=100,batch_size=256,learning_rate=0.001,
                 steps=4,trace="hutchinson",seed=1103,device="auto"):
        if min(window,hidden,epochs,batch_size,steps)<1 or trace not in {"exact","hutchinson"}:
            raise ValueError("Positive dimensions/budgets and valid trace required")
        self.window,self.hidden,self.epochs=window,hidden,epochs
        self.batch_size,self.learning_rate,self.steps,self.trace=batch_size,learning_rate,steps,trace
        self.seed=seed
        self.device=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device)

    def fit(self,windows):
        x=_windows(windows,self.window).flatten(1)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=="cuda" else []):
            torch.manual_seed(self.seed)
            self.forward_field=TimeMLP(x.shape[1],self.hidden).to(self.device)
            self.reverse_field=TimeMLP(x.shape[1],self.hidden).to(self.device)
            optimizer=torch.optim.Adam(list(self.forward_field.parameters())+list(self.reverse_field.parameters()),lr=self.learning_rate)
            self.loss_history_=[]
            for _ in range(self.epochs):
                total=0.
                for ids in torch.randperm(len(x)).split(self.batch_size):
                    target=x[ids].to(self.device); prior=torch.randn_like(target); t=torch.rand(len(ids),device=self.device)
                    loss=dual_cosine_loss(self.forward_field(t,target),self.reverse_field(t,prior))
                    optimizer.zero_grad(); loss.backward(); optimizer.step();total+=float(loss.detach())*len(ids)
                self.loss_history_.append(total/len(x))
        self.forward_field.eval(); self.reverse_field.eval()
        return self

    def score(self,windows):
        if not hasattr(self,"reverse_field"): raise RuntimeError("Call fit before score")
        x=_windows(windows,self.window).flatten(1)
        scores=[]; rng=torch.Generator(device=self.device).manual_seed(self.seed+1)
        for batch in x.split(self.batch_size):
            scores.append(-cnf_log_density(self.reverse_field,batch.to(self.device),self.steps,self.trace,rng).cpu().numpy())
        return np.concatenate(scores)


class PrismFlowCore(nn.Module):
    """Trainable Eq.5--14 core with gradient-isolated hard expert selection.

    Flat MLP encoder/backbone and lambda(t)=t are reconstruction choices; paper
    does not publish complete architecture/schedules. Conditional guidance and
    generation metric evaluators are outside this core.
    """
    def __init__(self,dimensions,hidden=128,latent=16,experts=4,delta=0.01,
                 beta=0.1,alpha_w=1.,alpha_b=0.1,gamma=1.,variant="full"):
        super().__init__()
        if variant not in {"full","vanilla_fm","no_wta","no_balance","beta_zero","gamma_zero"}:
            raise ValueError("Unknown PrismFlow variant")
        if min(dimensions,hidden,latent,experts)<1 or delta<0:
            raise ValueError("Positive dimensions and nonnegative dissipative margin required")
        self.variant,self.beta,self.alpha_w,self.alpha_b,self.gamma=variant,beta,alpha_w,alpha_b,gamma
        self.latent,self.experts,self.delta=latent,experts,delta
        self.global_field=TimeMLP(dimensions,hidden)
        self.encoder=nn.Sequential(nn.Linear(dimensions,hidden),nn.ReLU(),nn.Linear(hidden,latent))
        self.router=nn.Linear(latent+1,experts)
        self.S=nn.Parameter(torch.randn(experts,latent,latent)*0.01)
        self.R=nn.Parameter(torch.randn(experts,latent,latent)*0.01)
        self.decoder=nn.Sequential(nn.Linear(2*latent,hidden),nn.ReLU(),nn.Linear(hidden,dimensions))

    def generators(self):
        return self.S-self.S.transpose(-1,-2)-self.R.transpose(-1,-2)@self.R-self.delta*torch.eye(self.latent,device=self.S.device)

    def components(self,t,x):
        z=self.encoder(x); probs=self.router(torch.cat((z,t.reshape(-1,1)),-1)).softmax(-1)
        dz=torch.einsum("kij,bj->bki",self.generators(),z)
        residual=self.decoder(torch.cat((z[:,None,:].expand(-1,self.experts,-1),dz),-1))
        return self.global_field(t,x),residual,probs

    def loss(self,source,target,time):
        x=(1-time[:,None])*source+time[:,None]*target
        global_v,residual,probs=self.components(time,x)
        cfm=F.mse_loss(global_v,target-source)
        # Stop global gradients from specialization, as the paper isolates losses.
        endpoint=x[:,None,:]+(1-time[:,None,None])*(global_v.detach()[:,None,:]+time[:,None,None]*residual)
        beta=0. if self.variant=="beta_zero" else self.beta
        scores=(endpoint-target[:,None,:]).square().sum(-1)-beta*probs.clamp_min(1e-8).log()
        winner=scores.detach().argmin(-1)
        wta=(time*scores.gather(1,winner[:,None]).squeeze(1)).mean()
        uniform=1/self.experts
        balance=(uniform*(math.log(uniform)-probs.mean(0).clamp_min(1e-8).log())).sum()
        if self.variant=="vanilla_fm": return cfm
        return cfm+(0 if self.variant=="no_wta" else self.alpha_w)*wta+(0 if self.variant=="no_balance" else self.alpha_b)*balance

    def forward(self,time,x):
        global_v,residual,probs=self.components(time,x)
        if self.variant in {"vanilla_fm","gamma_zero"}: return global_v
        winner=probs.argmax(-1)
        selected=residual[torch.arange(len(x),device=x.device),winner]
        return global_v+self.gamma*time[:,None]*selected

    def sample(self,source,steps=100):
        if steps<1: raise ValueError("Need positive step count")
        x=source.clone()
        with torch.no_grad():
            for i in range(steps):
                t=torch.full((len(x),),i/steps,device=x.device,dtype=x.dtype)
                x=x+self(t,x)/steps
        return x


class PrismFlowModel:
    """Fit/sample wrapper for the explicitly limited PrismFlow core."""
    def __init__(self,window=24,hidden=128,latent=16,experts=4,delta=0.01,
                 beta=0.1,alpha_w=1.,alpha_b=0.1,gamma=1.,variant="full",
                 epochs=100,batch_size=256,learning_rate=0.001,steps=100,seed=1103,device="cpu"):
        self.window,self.epochs,self.batch_size=window,epochs,batch_size
        self.learning_rate,self.steps,self.seed,self.device=learning_rate,steps,seed,torch.device(device)
        self.core_parameters=dict(hidden=hidden,latent=latent,experts=experts,delta=delta,
                                  beta=beta,alpha_w=alpha_w,alpha_b=alpha_b,gamma=gamma,variant=variant)

    def fit(self,windows):
        x=_windows(windows,self.window);self.channels=x.shape[-1];x=x.flatten(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model=PrismFlowCore(x.shape[1],**self.core_parameters).to(self.device)
            optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
            self.loss_history_=[]
            for _ in range(self.epochs):
                total=0.
                for ids in torch.randperm(len(x)).split(self.batch_size):
                    target=x[ids].to(self.device);source=torch.randn_like(target);t=torch.rand(len(ids),device=self.device)
                    loss=self.model.loss(source,target,t)
                    optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
                self.loss_history_.append(total/len(x))
        self.model.eval()
        return self

    def sample(self,count,seed=None):
        if not hasattr(self,"model"): raise RuntimeError("Call fit first")
        if count<1: raise ValueError("Positive sample count required")
        rng=torch.Generator(device=self.device).manual_seed(self.seed+1 if seed is None else seed)
        source=torch.randn((count,self.window*self.channels),generator=rng,device=self.device)
        return self.model.sample(source,self.steps).reshape(count,self.window,self.channels).cpu().numpy()
