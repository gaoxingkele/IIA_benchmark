"""Paper-derived supplemental primitives, not certified full reproductions.

JFI: CES 2026, DOI 10.1016/j.ces.2026.124467, pp. 5-7/Table 1.
Flow Mismatching: arXiv:2605.23070v1, Section 3, Eqs. (1)-(4).
Infini-Channel Mixer: arXiv:2409.13530v1, p. 3/Eqs. (1)-(2),
Appendix B, pp. 7-8. Configs document integration/experiment gaps.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def local_prior_mean(values: Tensor, observed: Tensor) -> Tensor:
    """JFI Eq. (9): interpolate observed samples; nearest ends; zero empty.

    Shapes are [batch,time,feature]. Interpolation uses only observed entries;
    callers must pass windows available at the current deployment timestamp.
    """
    if values.ndim != 3 or observed.shape != values.shape:
        raise ValueError("expected equal [batch,time,feature] shapes")
    mask = observed.bool()
    if not torch.isfinite(values[mask]).all():
        raise ValueError("observed values must be finite")
    mean = torch.zeros_like(values)
    length = values.shape[1]
    for b in range(values.shape[0]):
        for j in range(values.shape[2]):
            inds = mask[b, :, j].nonzero().flatten()
            if inds.numel() == 0:
                continue
            idx = torch.arange(length, device=values.device)
            pos = torch.searchsorted(inds, idx)
            left = inds[(pos - 1).clamp(0, inds.numel() - 1)]
            right = inds[pos.clamp(0, inds.numel() - 1)]
            alpha = (idx - left).to(values.dtype) / (right - left).clamp_min(1)
            mean[b, :, j] = values[b, left, j] * (1 - alpha) + values[b, right, j] * alpha
    return torch.where(mask, values, mean)


def sample_jfi_prior(values: Tensor, observed: Tensor, *, variance: float = 0.01,
                     local: bool = True, generator: torch.Generator | None = None) -> Tensor:
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    mean = local_prior_mean(values, observed) if local else torch.zeros_like(values)
    noise = torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator)
    sampled = mean + noise * (math.sqrt(variance) if local else 1.0)
    return torch.where(observed.bool(), values, sampled)


class _MixedConv(nn.Module):
    def __init__(self, inputs: int, outputs: int, mixed: bool):
        super().__init__()
        if outputs % 4:
            raise ValueError("mixed channels must be divisible by four")
        self.mixed = mixed
        if mixed:
            branch = outputs // 4
            def path(kernel, padding, dilation=1):
                return nn.Sequential(nn.Conv2d(inputs, branch, 1),
                    nn.Conv2d(branch, branch, kernel, padding=padding, dilation=dilation, groups=branch))
            # Axes [time,feature]. The paper labels its 1x9 temporal branch
            # oppositely to this layout: retain the published kernel sizes.
            self.branches = nn.ModuleList([path(5, 4, 2), path((1, 9), (0, 4)),
                path((9, 1), (4, 0)), nn.Sequential(nn.Conv2d(inputs, branch, 1),
                    nn.Conv2d(branch, branch, 3, padding=1, groups=branch),
                    nn.Conv2d(branch, branch, 3, padding=1, groups=branch))])
        else:
            self.conv = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = torch.cat([p(x) for p in self.branches], 1) if self.mixed else self.conv(x)
        return F.relu(self.norm(x))


class _MCB(nn.Module):
    def __init__(self, inputs, outputs, time_dim, mixed):
        super().__init__()
        self.time = nn.Linear(time_dim, inputs)
        self.layers = nn.Sequential(_MixedConv(inputs, outputs, mixed), _MixedConv(outputs, outputs, mixed))

    def forward(self, x, t):
        return self.layers(x + self.time(t)[:, :, None, None])


class _ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, max(1, channels // 8)), nn.ReLU(),
                                  nn.Linear(max(1, channels // 8), channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.mean((-2, -1)))[:, :, None, None]


class JFIVelocity(nn.Module):
    """U-Net from Table 1, plus w/o MCB,CA switch.

    Published unspecified branch reduction/depthwise details are explicit local
    choices, so this is a paper-derived architecture, not author-equivalent code.
    Two MCBs per down/up stage follow Table 1 (each MCB has two submodules).
    """
    def __init__(self, width: int = 32, mixed: bool = True, channel_attention: bool = True):
        super().__init__()
        if width % 4 or width < 4:
            raise ValueError("width must be a positive multiple of four")
        self.width = width
        self.input = nn.Conv2d(1, width, 3, padding=1)
        self.down1 = nn.ModuleList([_MCB(width, width*2, width, mixed), _MCB(width*2, width*2, width, mixed)])
        self.down2 = nn.ModuleList([_MCB(width*2, width*4, width, mixed), _MCB(width*4, width*4, width, mixed)])
        self.middle = _MCB(width*4, width*4, width, mixed)
        self.up1 = nn.ModuleList([_MCB(width*8, width*2, width, mixed), _MCB(width*2, width*2, width, mixed)])
        self.up2 = nn.ModuleList([_MCB(width*4, width, width, mixed), _MCB(width, width, width, mixed)])
        self.ca1 = _ChannelAttention(width*2) if channel_attention else nn.Identity()
        self.ca2 = _ChannelAttention(width*4) if channel_attention else nn.Identity()
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        if state.ndim != 3 or min(state.shape[1:]) < 4:
            raise ValueError("state must be [batch,time>=4,feature>=4]")
        if time.numel() != state.shape[0]:
            raise ValueError("one flow time per batch sample required")
        freq = torch.exp(-math.log(10000) * torch.arange(self.width//2, device=state.device, dtype=state.dtype)
                         / max(1, self.width//2 - 1))
        angles = time.reshape(-1, 1) * freq
        emb = torch.cat([angles.sin(), angles.cos()], -1)
        x = self.input(state.unsqueeze(1))
        for layer in self.down1: x = layer(x, emb)
        skip1 = x
        x = F.max_pool2d(x, 2)
        for layer in self.down2: x = layer(x, emb)
        skip2 = x
        x = self.middle(F.max_pool2d(x, 2), emb)
        x = torch.cat([F.interpolate(x, size=skip2.shape[-2:], mode='nearest'), self.ca2(skip2)], 1)
        for layer in self.up1: x = layer(x, emb)
        x = torch.cat([F.interpolate(x, size=skip1.shape[-2:], mode='nearest'), self.ca1(skip1)], 1)
        for layer in self.up2: x = layer(x, emb)
        return self.output(x).squeeze(1)


def jfi_objective(velocity: Callable, truth: Tensor, observed: Tensor, *, sigma: float = 0.,
                  prior_variance: float = 0.01, local_prior: bool = True,
                  joint: bool = True, target_weight: float = 1., distillation_weight: float = 1.,
                  teacher: Callable | None = None, task: str = 'regression', temperature: float = 1.,
                  generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """CFM Eq.7 + one-step Eq.12 + forward KL Eq.13/14 or MSE.

    Frozen teacher parameters are the caller's responsibility. Teacher truth
    outputs are detached; generated-sample gradients are retained.
    """
    if not 0 <= sigma < 1 or temperature <= 0 or target_weight < 0 or distillation_weight < 0:
        raise ValueError("invalid sigma/temperature/loss weights")
    if joint and distillation_weight > 0 and teacher is None:
        raise ValueError("joint distillation requires a frozen downstream teacher; set weight=0 explicitly to omit it")
    if not torch.isfinite(truth).all():
        raise ValueError("CFM training requires complete finite ground truth windows")
    z0 = sample_jfi_prior(truth, observed, variance=prior_variance, local=local_prior, generator=generator)
    time = torch.rand(truth.shape[0], device=truth.device, dtype=truth.dtype, generator=generator)
    t = time[:, None, None]
    zt = (1-(1-sigma)*t)*z0 + t*truth
    # Eq.7 uses the full vector field, even though Eq.8 masks sampling.
    flow = F.mse_loss(velocity(zt, time), truth-(1-sigma)*z0)
    generated = z0 + velocity(z0, torch.zeros_like(time)) * (~observed.bool())
    target = F.mse_loss(generated, truth)
    kd = truth.new_zeros(())
    if joint and teacher is not None:
        reference = teacher(truth).detach()
        prediction = teacher(generated)
        if task == 'classification':
            log_p = F.log_softmax(prediction/temperature, -1)
            log_q = F.log_softmax(reference/temperature, -1)
            kd = temperature**2 * (log_p.exp()*(log_p-log_q)).sum(-1).mean()
        elif task == 'regression': kd = F.mse_loss(prediction, reference)
        else: raise ValueError("task must be classification or regression")
    loss = flow + (target_weight*target + distillation_weight*kd if joint else 0)
    return dict(loss=loss, flow=flow, target=target, distillation=kd)


@torch.no_grad()
def jfi_impute(velocity: Callable, values: Tensor, observed: Tensor, *, steps: int = 1,
               local_prior: bool = True, prior_variance: float = 0.01,
               generator: torch.Generator | None = None) -> Tensor:
    """Masked Euler Eq.8; reconstructs observed entries after every step."""
    if steps < 1: raise ValueError("steps must be positive")
    state = sample_jfi_prior(values, observed, local=local_prior, variance=prior_variance, generator=generator)
    for step in range(steps):
        time = state.new_full((state.shape[0],), step/steps)
        state = state + velocity(state, time) * (~observed.bool()) / steps
        state = torch.where(observed.bool(), values, state)
    return state


@torch.no_grad()
def flow_mismatch(velocity: Callable, images: Tensor, *, paths: int = 8, times: int = 8,
                  aggregation: str = 'minimum', percentile: float = 0.1,
                  weighted: bool = True, top_fraction: float = 0.01,
                  seeds: Tensor | None = None, generator: torch.Generator | None = None,
                  labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Image-only Eq.2-4; interior times, channel squared norm, top1% score.

    Percentile is the published ceil(alpha*K) order statistic, not interpolated
    quantile. Explicit seeds enable paired ablations. Does not alter model mode.
    """
    if images.ndim != 4 or paths < 1 or times < 1 or not 0 < top_fraction <= 1:
        raise ValueError("expected [batch,channel,height,width] and positive counts")
    if aggregation not in ('minimum','average','percentile') or not 0 < percentile <= 1:
        raise ValueError("invalid path aggregator")
    shape = (paths,) + tuple(images.shape)
    if seeds is None: seeds = torch.randn(shape, dtype=images.dtype, device=images.device, generator=generator)
    if seeds.shape != shape: raise ValueError("seed shape mismatch")
    if labels is not None and labels.shape!=(images.shape[0],):
        raise ValueError("one category label per image required")
    target = images.unsqueeze(0).expand_as(seeds)
    geometric = target-seeds
    heat = images.new_zeros(images.shape[0], *images.shape[-2:])
    for j in range(1, times+1):
        t = j/(times+1)
        state = ((1-t)*seeds+t*target).flatten(0,1)
        time=state.new_full((state.shape[0],),t)
        predicted=(velocity(state,time,labels.repeat(paths)) if labels is not None
                   else velocity(state,time)).reshape_as(seeds)
        residual = (predicted-geometric).square().sum(2)
        if aggregation == 'minimum': delta = residual.min(0).values
        elif aggregation == 'average': delta = residual.mean(0)
        else: delta = residual.kthvalue(max(1, math.ceil(percentile*paths)), dim=0).values
        heat += delta * (t*t if weighted else 1) / times
    flat = heat.flatten(1)
    top_n = max(1, math.ceil(flat.shape[1]*top_fraction))
    score = flat.max(1).values + flat.topk(top_n, dim=1).values.mean(1)
    return heat, score


def image_cfm_objective(velocity: Callable, images: Tensor, *, generator=None,
                        labels: Tensor | None = None) -> Tensor:
    """Normal-image CFM training Eq.1; model architecture supplied by caller."""
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    time = torch.rand(images.shape[0], device=images.device, dtype=images.dtype, generator=generator)
    t = time.reshape((-1,)+(1,)*(images.ndim-1))
    if labels is not None and labels.shape!=(images.shape[0],):
        raise ValueError("one category label per image required")
    state=(1-t)*noise+t*images
    prediction=velocity(state,time,labels) if labels is not None else velocity(state,time)
    return F.mse_loss(prediction, images-noise)


class InfiniChannelMixer(nn.Module):
    """ICM Eqs.1-2 on projected Q,K,V [B,channels,heads,tokens,key_dim].

    No state persists between independent samples. One beta per attention head.
    Reusing backbone projections/T5 biases must be done by integration caller.
    """
    def __init__(self, heads: int, *, mode: str = 'icm', train_beta: bool = True,
                  epsilon: float = 1e-6):
        super().__init__()
        if mode not in ('icm','independent','concatenation') or heads < 1 or epsilon <= 0:
            raise ValueError("invalid attention configuration")
        self.beta = nn.Parameter(torch.zeros(heads), requires_grad=train_beta)
        self.mode, self.epsilon = mode, epsilon
        self.inter_bias = nn.Parameter(torch.zeros(2)) if mode == 'concatenation' else None

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if query.ndim != 5 or query.shape != key.shape or key.shape != value.shape or query.shape[2] != self.beta.numel():
            raise ValueError("expected equal [B,C,H,N,D] Q/K/V tensors")
        if self.mode == 'concatenation':
            b,c,h,n,d = query.shape
            q,k,v = [x.permute(0,2,1,3,4).reshape(b,h,c*n,d) for x in (query,key,value)]
            channels = torch.arange(c,device=q.device).repeat_interleave(n)
            same = channels[:,None] == channels[None,:]
            bias = torch.where(same,self.inter_bias[0],self.inter_bias[1])
            out = ((q@k.transpose(-2,-1)/math.sqrt(d)+bias).softmax(-1)@v)
            return out.reshape(b,h,c,n,d).permute(0,2,1,3,4)
        local = (query@key.transpose(-2,-1)/math.sqrt(query.shape[-1])).softmax(-1)@value
        if self.mode == 'independent': return local
        projected_key, projected_query = F.elu(key)+1, F.elu(query)+1
        memory = torch.einsum('bchnd,bchne->bhde',projected_key,value)
        normalizer = projected_key.sum((1,3))
        retrieved = torch.einsum('bchnd,bhde->bchne',projected_query,memory)
        retrieved = retrieved/(torch.einsum('bchnd,bhd->bchn',projected_query,normalizer).unsqueeze(-1)+self.epsilon)
        gate = self.beta.sigmoid()[None,None,:,None,None]
        return gate*retrieved+(1-gate)*local


class StaticChannelEmbedding(nn.Module):
    """Appendix B.2 static-channel variant for [B,C,N,D] embeddings."""
    def __init__(self, channels: int, dimension: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(channels,dimension))
        nn.init.normal_(self.embedding,std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != self.embedding.shape[0] or tokens.shape[-1] != self.embedding.shape[1]:
            raise ValueError("channel embedding shape mismatch")
        return tokens+self.embedding[None,:,None,:]
