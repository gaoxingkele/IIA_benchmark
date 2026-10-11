"""Full-image WT reverse-FM training in Flow Mismatching AppendixC2/Eqs21–23.

Related native source: https://github.com/lil-wayne-0319/fmad .
This controlled image adaptation does not certify the original feature method.
"""
import torch
from .fm_reflect_wt_native import wt_native_objective


def wt_uniform_objective(velocity, images, *, native_core, epsilon, labels=None, generator=None):
    """Independent uniform time then full Gaussian; original native draw ordering."""
    time=torch.rand(images.shape[0],device=images.device,dtype=images.dtype,generator=generator)
    noise=torch.randn(images.shape,device=images.device,dtype=images.dtype,generator=generator)
    return wt_native_objective(velocity,images,time,noise,native_core=native_core,
        epsilon=epsilon,labels=labels)
