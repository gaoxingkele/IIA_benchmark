"""TCCM image-domain adaptation in arXiv:2605.23070, p22/Eqs19-20.

This is the controlled comparison described by Flow Mismatching, not a claim
to reproduce TCCM's best original domain/backbone setting.
"""
import torch
from .fm_image_comparison import tccm_objective, tccm_heatmap


def tccm_uniform_objective(velocity, images, *, labels=None, generator=None):
    """Uniform unit time, clean input and contraction target -x; no FM noise."""
    time = torch.rand(images.shape[0], device=images.device, dtype=images.dtype, generator=generator)
    return tccm_objective(velocity, images, time, labels=labels)


def score_tccm_image(velocity, images, *, labels=None, top_fraction=.01):
    """Eq20 unsquared channel norm at t1; shared local max-plus-top1% scalar."""
    import math
    if not 0 < top_fraction <= 1:
        raise ValueError('Registered positive image-score top fraction required')
    heat = tccm_heatmap(velocity, images, labels=labels)
    flat = heat.flatten(1)
    top = max(1, math.ceil(flat.shape[1] * top_fraction))
    score = flat.max(1).values + flat.topk(top, dim=1).values.mean(1)
    return heat, score
