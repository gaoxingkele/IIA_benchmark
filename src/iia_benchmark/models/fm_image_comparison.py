"""Flow Mismatching Appendix C.2 equation components, not certified runs.

The supplied velocity, masks, WT statistics and ODE solver are explicit.
Undocumented mask/solver choices must not silently become paper defaults.
Source: https://arxiv.org/abs/2605.23070, pp. 21--23, Eqs. 14--29.
"""
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


def _images(images: Tensor) -> None:
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError('floating [batch,channel,height,width] images required')


def _call(velocity: Callable, images: Tensor, time: Tensor,
          labels: Tensor | None) -> Tensor:
    if labels is not None and labels.shape != (images.shape[0],):
        raise ValueError('one category label per image required')
    result = velocity(images, time, labels) if labels is not None else velocity(images, time)
    if result.shape != images.shape:
        raise ValueError('velocity field shape differs from image shape')
    return result


def reflect_corruption(clean: Tensor, mask: Tensor, *, alpha: Tensor,
                       beta: Tensor, spatial_noise: Tensor, scalar_noise: Tensor) -> Tensor:
    """Eqs. 14--15. Caller supplies the original random-walk mask and draws.

    Mask is binary and broadcastable to images; alpha/beta lie in [0,1].
    Draws are explicit because mask count/length/connectivity and draw sharing
    were not specified sufficiently to certify the full author's training.
    """
    _images(clean)
    if spatial_noise.shape != clean.shape:
        raise ValueError('spatial noise must have the complete image shape')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('Reflect mask must be binary')
    for value in (alpha, beta):
        if not bool(((value >= 0) & (value <= 1)).all()):
            raise ValueError('mixing draws must lie in [0,1]')
    replacement = beta.sqrt() * scalar_noise + (1-beta).sqrt() * spatial_noise
    altered = (1-alpha).sqrt()*clean + alpha.sqrt()*replacement
    return torch.where(mask.bool(), altered, clean)


def reflect_objective(velocity: Callable, clean: Tensor, corrupted: Tensor,
                      time: Tensor, *, labels: Tensor | None = None) -> Tensor:
    """Eq. 16, with caller's corruptions and uniform training-time draws."""
    _images(clean)
    if corrupted.shape != clean.shape or time.shape != (clean.shape[0],):
        raise ValueError('paired image shapes and one time per image required')
    t = time[:, None, None, None]
    return F.mse_loss(_call(velocity, (1-t)*corrupted+t*clean, time, labels), clean-corrupted)


@torch.no_grad()
def reflect_heatmap(velocity: Callable, images: Tensor, *, labels: Tensor | None = None) -> Tensor:
    """Eq. 17 and following norm heatmap; test residual is NOT squared."""
    _images(images)
    return torch.linalg.vector_norm(_call(velocity, images, images.new_zeros(images.shape[0]), labels), dim=1)


def tccm_objective(velocity: Callable, images: Tensor, time: Tensor,
                   *, labels: Tensor | None = None) -> Tensor:
    """Eq. 19, contraction target -x, with caller's uniform time draws."""
    _images(images)
    if time.shape != (images.shape[0],):
        raise ValueError('one training time per image required')
    return F.mse_loss(_call(velocity, images, time, labels), -images)


@torch.no_grad()
def tccm_heatmap(velocity: Callable, images: Tensor, *, labels: Tensor | None = None) -> Tensor:
    """Eq. 20, unsquared contraction residual at the published fixed t=1."""
    _images(images)
    residual = _call(velocity, images, images.new_ones(images.shape[0]), labels) + images
    return torch.linalg.vector_norm(residual, dim=1)


def wt_affine(images: Tensor, *, mean: Tensor, variance: Tensor, epsilon: float) -> Tensor:
    """Eq. 21; caller explicitly supplies per-sample WT moments and epsilon.

    The paper does not pin epsilon or the variance estimator. Passing moments
    avoids pretending a population/sample variance choice is author verified.
    """
    _images(images)
    if mean.shape != (images.shape[0], 1, 1, 1) or variance.shape != mean.shape:
        raise ValueError('one broadcastable mean and variance per image required')
    if epsilon <= 0 or not bool((variance >= 0).all()):
        raise ValueError('positive epsilon and nonnegative variance required')
    return (images-mean) * torch.rsqrt(variance+epsilon)


def wt_flow_objective(velocity: Callable, mapped_images: Tensor, noise: Tensor,
                      time: Tensor, *, labels: Tensor | None = None) -> Tensor:
    """Eqs. 22--23: normal WT-mapped image to Gaussian, reverse direction."""
    _images(mapped_images)
    if noise.shape != mapped_images.shape or time.shape != (mapped_images.shape[0],):
        raise ValueError('full image-shaped noise and one training time per image required')
    t = time[:, None, None, None]
    return F.mse_loss(_call(velocity, (1-t)*mapped_images+t*noise, time, labels), noise-mapped_images)


@torch.no_grad()
def wt_flow_heatmap(mapped_images: Tensor, integrate: Callable, *, labels: Tensor | None = None) -> Tensor:
    """Eqs. 24--25; integrate(state,t_start,t_end,labels) supplies native solver."""
    _images(mapped_images)
    endpoint = integrate(mapped_images, 0.0, 1.0, labels)
    if endpoint.shape != mapped_images.shape:
        raise ValueError('ODE endpoint shape differs from full image shape')
    return endpoint.square().sum(1) / 2


@torch.no_grad()
def reconflow_heatmap(images: Tensor, noise: Tensor, integrate: Callable,
                      *, labels: Tensor | None = None) -> Tensor:
    """Eqs. 28--29, t_init=.5 and K=10; integrate must implement native T=10.

    The ODE scheme remains a reproduction gap; it is not replaced with Euler.
    Returns the minimum per-pixel channel L2 residual over ten reconstructions.
    """
    _images(images)
    if noise.shape != (10,) + tuple(images.shape):
        raise ValueError('original ReconFlow requires ten full image-shaped noise paths')
    if labels is not None and labels.shape != (images.shape[0],):
        raise ValueError('one category label per image required')
    target = images.unsqueeze(0).expand_as(noise)
    state = (.5*noise+.5*target).flatten(0, 1)
    endpoint = integrate(state, .5, 1.0, labels.repeat(10) if labels is not None else None)
    if endpoint.shape != state.shape:
        raise ValueError('ODE endpoint shape differs from complete path batch')
    return torch.linalg.vector_norm(target-endpoint.reshape_as(noise), dim=2).min(0).values
