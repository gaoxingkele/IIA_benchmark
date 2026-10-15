"""Official fixed512 MOMENT inference with native masks for short raw series.

Zeroes used to fill tensor storage are never marked as observed or scored.
For full contexts the unchanged original reconstruction implementation is used.
"""
from __future__ import annotations
import numpy as np
import torch
from .fm_moment_pretrained import MomentPretrainedWindow,reconstruct_windows


class MaskedContextWindows(np.ndarray):
    """One complete short source, with an explicit count of real time points."""
    def __new__(cls,values,window=512,fill_value=0.):
        values=np.asarray(values)
        if (values.ndim!=2 or not 8<=len(values)<window or window!=512
                or not np.isfinite(values).all() or not np.isfinite(fill_value)):
            raise ValueError('Finite complete short source and original512 context required')
        storage=np.full((1,window,values.shape[1]),fill_value,dtype=values.dtype)
        storage[0,:len(values)]=values
        result=storage.view(cls);result.valid_length=len(values)
        return result

    def __array_finalize__(self,source):
        self.valid_length=getattr(source,'valid_length',None)


def reconstruct_masked_context(model,windows,batch_size,device):
    if (not isinstance(windows,MaskedContextWindows) or windows.ndim!=3
            or windows.shape[0]!=1 or windows.shape[1]!=512
            or not isinstance(windows.valid_length,int) or not 8<=windows.valid_length<512
            or not np.isfinite(windows).all() or batch_size!=1):
        raise ValueError('One full short source with explicit real-point metadata required')
    length=windows.valid_length
    inputs=torch.from_numpy(np.asarray(windows).astype(np.float32,copy=False)).to(device).transpose(1,2)
    observed=torch.zeros((1,512),device=device);observed[:,:length]=1
    with torch.inference_mode():
        output=model(x_enc=inputs,input_mask=observed,mask=observed).reconstruction
        original=inputs[...,:length];reconstructed=output[...,:length]
        if output.shape!=inputs.shape or not torch.isfinite(reconstructed).all():
            raise ValueError('All real source points require finite reconstructions')
        return (original-reconstructed).square().mean(dim=1).cpu().numpy()


class MomentMaskedContextWindow(MomentPretrainedWindow):
    def __init__(self,*args,observation_protocol='right_structural_fill_native_mask_real_targets_v1',**kwargs):
        super().__init__(*args,**kwargs)
        if (self.window!=512 or self.batch_size!=1 or self.padding_policy!='minimal_patch'
                or observation_protocol!='right_structural_fill_native_mask_real_targets_v1'):
            raise ValueError('Original512 architecture/batch1 and declared native mask required')
        self.observation_protocol=observation_protocol

    def score(self,windows):
        if not hasattr(self,'model'):raise RuntimeError('Official pretrained weights not loaded')
        if isinstance(windows,MaskedContextWindows):
            return reconstruct_masked_context(self.model,windows,self.batch_size,self.device)
        if windows.ndim!=3 or windows.shape[1]!=512:raise ValueError('Original512 context required')
        return reconstruct_windows(self.model,windows,self.batch_size,self.device,self.padding_policy)
