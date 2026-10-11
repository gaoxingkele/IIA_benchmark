"""Related native WT endpoint, target Appendix C2 Eq25 image energy.

Refs: arxiv.org/abs/2605.23070 and github.com/lil-wayne-0319/fmad.
Explicit source/epsilon/step settings do not certify the target author's solver.
"""
import math
import torch
from .fm_reflect_wt_native import wt_native_standardize, wt_native_endpoint


@torch.no_grad()
def score_wt_image(velocity, images, *, native_core, epsilon, steps, top_fraction, labels=None):
    if not 0 < top_fraction <= 1:
        raise ValueError('Explicit positive top-pixel fraction required')
    mapped = wt_native_standardize(images, epsilon=epsilon)
    endpoint = wt_native_endpoint(velocity, mapped, native_core=native_core, steps=steps, labels=labels)
    heatmap = .5 * endpoint.square().sum(dim=1)
    flat = heatmap.flatten(1)
    count = max(1, math.ceil(flat.shape[1] * top_fraction))
    score = flat.max(dim=1).values + flat.topk(count, dim=1).values.mean(dim=1)
    return heatmap, score
