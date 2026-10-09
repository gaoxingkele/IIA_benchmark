"""Thin adapters for cached, paired author data and independent sampling fields.

Author model, loss, optimizer, EMA, solver and training loop stay in their pinned
source tree. Paths and all budgets are provided by registered configurations.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def build_author_model(source_root,model_target,parameters):
    import importlib
    import sys
    sys.path.insert(0,str(source_root))
    module,name=model_target.rsplit('.',1)
    return getattr(importlib.import_module(module),name)(**parameters)


class FrozenAuthorGenerationDataset(Dataset):
    def __init__(self, path):
        self.samples=np.load(path,mmap_mode='r',allow_pickle=False)
        if self.samples.ndim!=3 or not np.isfinite(self.samples).all():
            raise ValueError('Frozen original full generation data must be finite N,T,D')
        self.sample_num,self.window,self.var_num=self.samples.shape
        self.auto_norm=True

    def __getitem__(self,index):
        return torch.from_numpy(self.samples[index].copy()).float()

    def __len__(self):
        return self.sample_num


def chunked_field(field,x,t,chunk_size):
    """Preserve original sample order and noise; chunk only independent fields."""
    if chunk_size<1 or len(x)!=len(t):
        raise ValueError('Positive chunk size and matched time batch required')
    values=[]; aux={};track_graph=torch.is_grad_enabled()
    for start in range(0,len(x),chunk_size):
        stop=min(start+chunk_size,len(x))
        value,observations=field(x[start:stop],t[start:stop])
        # The original field enables autograd internally to compute d/dx.
        # Sampling does not need the resulting higher-order graph. Release each
        # chunk's graph before evaluating the next chunk, retaining full values.
        values.append(value if track_graph else value.detach())
        for key,observation in observations.items():
            aux[key]=aux.get(key,0.)+float(observation)*(stop-start)/len(x)
    return torch.cat(values,dim=0),aux


def bind_chunked_sampling(model,chunk_size):
    original=model.compute_flow
    def field(x,t):
        return chunked_field(original,x,t,chunk_size)
    model.compute_flow=field
    return original
