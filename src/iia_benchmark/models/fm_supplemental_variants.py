"""Supplemental ablations with explicit optimization and reconstruction semantics.

SensitiveHUE pp. 4-6, Table 4, Appendix B.3; beta-NLL Eq.8.
Faithful: Stirn et al. AISTATS 2023, Eq.5 (stop mean and shared trunk).
Natural: Immer et al. ICML 2023, Sec.3.2/Eq.2 (Gaussian natural heads).
Original SensitiveHUE Table4 experiment scripts are not publicly supplied.
These are formula-derived variants, not certified equivalents of those runs.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fm_supplemental_author import load_sensitivehue


def hue_variant_loss(mean: Tensor, target: Tensor, log_precision: Tensor, *,
                     mode: str = 'mts_nll', alpha: float = 1., beta: float = .5,
                     loss_mask: Tensor | None = None) -> Tensor:
    """Gaussian objectives with the paper's 1/2 factor (author release omits it).

    Faithful requires a model which detaches the variance-head trunk input;
    detaching just the mean here is only one of its two required modifications.
    Natural requires eta1/log-precision model heads, not just a relabeled loss.
    """
    if mean.shape != target.shape or log_precision.shape != target.shape or mean.ndim != 3:
        raise ValueError('expected equal [batch,time,feature] tensors')
    if mode not in ('mse','nll','mts_nll','beta_nll','faithful','natural') or not 0 <= beta <= 1 or not 0 <= alpha <= 1:
        raise ValueError('invalid objective mode or weighting coefficient')
    selected=torch.ones_like(mean,dtype=torch.bool) if loss_mask is None else loss_mask.bool()
    if selected.ndim==2: selected=selected[:,:,None].expand_as(mean)
    if selected.shape!=mean.shape or not selected.any():raise ValueError('loss mask has no valid entries or wrong shape')
    error=(mean-target).square()
    if mode=='mse':loss=.5*error
    elif mode=='faithful':
        variance_loss=.5*((mean.detach()-target).square()*log_precision.exp()-log_precision)
        loss=.5*error+variance_loss
    else:
        loss=.5*(error*log_precision.exp()-log_precision)
        if mode in ('beta_nll','mts_nll'):
            variance=(-log_precision).exp().detach()
            weight=variance.pow(beta if mode=='beta_nll' else 1.)
            if mode=='mts_nll':
                counts=selected.sum((0,1)).clamp_min(1)
                average=torch.where(selected,variance,0).sum((0,1))/counts
                weight=weight/average.pow(alpha)
            loss=weight*loss
    return loss[selected].mean()


class SensitiveHUEAblation(nn.Module):
    """Actual Transformer reconstruction with Table4 reconstruction/loss choices.

    SFR here follows Eq.3 and Eq.15 exactly, without released output-statistics
    rescaling. `revin` restores original input statistics. Compression uses the
    configured feature bottleneck. Mask zeros a sampled subset of complete
    timestamps across all channels, as Appendix B.3 describes.
    """
    def __init__(self, source_root: str, *, step_num_in: int, f_in: int,
                 dim_model: int = 128, head_num: int = 4, dim_hidden_fc: int = 256,
                 encode_layer_num: int = 1, dropout: float = .1,
                 strategy: str = 'sfr', loss_mode: str = 'mts_nll', alpha: float = 1.,
                 beta: float = .5, mask_ratio: float = .2, epsilon: float = 1e-5,
                 asset_root: str | None = None):
        super().__init__()
        if strategy not in ('sfr','revin','compression','mask','none'):
            raise ValueError('invalid reconstruction strategy')
        if loss_mode not in ('mts_nll','nll','beta_nll','faithful','natural','mse'):
            raise ValueError('invalid loss mode')
        if not 0 < mask_ratio < 1 or epsilon <= 0:raise ValueError('invalid mask ratio/epsilon')
        self.model=load_sensitivehue(source_root,step_num_in=step_num_in,f_in=f_in,
            dim_model=dim_model,head_num=head_num,dim_hidden_fc=dim_hidden_fc,
            encode_layer_num=encode_layer_num,dropout=dropout,statistical_removal=False,asset_root=asset_root)
        self.strategy,self.loss_mode,self.alpha,self.beta=strategy,loss_mode,alpha,beta
        self.mask_ratio,self.epsilon=mask_ratio,epsilon

    def create_mask(self, values: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """True selects withheld timestamps; exact count with shared channel mask."""
        count=max(1,int(values.shape[1]*self.mask_ratio))
        mask=torch.zeros(values.shape[:2],device=values.device,dtype=torch.bool)
        for b in range(values.shape[0]):
            positions=torch.randperm(values.shape[1],device=values.device,generator=generator)[:count]
            mask[b,positions]=True
        return mask

    def forward(self, values: Tensor, mask: Tensor | None = None,
                generator: torch.Generator | None = None) -> tuple[Tensor,Tensor]:
        if values.ndim!=3 or values.shape[2]!=self.model.f_in or not torch.isfinite(values).all():
            raise ValueError('expected finite [batch,time,feature] input')
        mean=values.mean(1,keepdim=True).detach()
        std=(values.var(1,unbiased=False,keepdim=True)+self.epsilon).sqrt().detach()
        source=values
        if self.strategy in ('sfr','revin'):source=(values-mean)/std
        elif self.strategy=='mask':
            if mask is None:mask=self.create_mask(values,generator)
            if mask.shape!=values.shape[:2]:raise ValueError('mask must select timestamps [batch,time]')
            source=values.masked_fill(mask.bool()[:,:,None],0)
        h=self.model.in_linear(source)+self.model.pos_embed(torch.arange(values.shape[1],device=values.device))
        for encoder in self.model.encoder:h,_=encoder(h,h)
        mean_output=self.model.rec_linear(h)
        # Faithful Eq.5 blocks the covariance contribution to shared parameters.
        log_precision=self.model.sigma_linear(h.detach() if self.loss_mode=='faithful' else h)
        if self.loss_mode=='natural':
            # eta1 = first linear head; eta2 = -.5 exp(second head).
            mean_output=mean_output*torch.exp(-log_precision)
        if self.strategy=='revin':mean_output=mean_output*std+mean
        return mean_output,log_precision

    def objective(self, target: Tensor, predictions: tuple[Tensor,Tensor],
                  mask: Tensor | None = None) -> Tensor:
        if self.strategy=='mask' and mask is None:
            raise ValueError('masked reconstruction loss requires the same explicit timestamp mask as forward')
        return hue_variant_loss(predictions[0],target,predictions[1],mode=self.loss_mode,
                                alpha=self.alpha,beta=self.beta,loss_mask=mask if self.strategy=='mask' else None)
