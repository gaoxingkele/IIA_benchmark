"""Load pinned SensitiveHUE model without altering its immutable source tree.

Source path is required from configs. No dataset or evaluation threshold is
selected here. Training loss follows the pinned author trainer.loss_func.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import Tensor, nn


class _NoRemoval(nn.Module):
    def forward(self, x: Tensor, mode: str) -> Tensor:
        return x


def _resolve_asset(source_root: str | Path, asset_root: str | Path | None) -> Path:
    """Resolve configured relative source paths against repo, never process cwd.

    Installed-package callers should pass asset_root or an absolute source_root.
    The repository checkout default is inferred from this module's location.
    """
    path=Path(source_root)
    if path.is_absolute(): return path.resolve()
    base=Path(asset_root).resolve() if asset_root is not None else Path(__file__).resolve().parents[3]
    return (base/path).resolve()


def load_sensitivehue(source_root: str | Path, *, step_num_in: int, f_in: int,
                      dim_model: int = 128, head_num: int = 4,
                      dim_hidden_fc: int = 256, encode_layer_num: int = 1,
                      dropout: float = 0.1, statistical_removal: bool = True,
                      asset_root: str | Path | None = None) -> nn.Module:
    """Load author model; optional SFR-removal ablation bypasses RevIN.

    This opt-out is local glue, not a supplied author ablation experiment script.
    Import caches are namespaced by resolved path to avoid unrelated packages.
    """
    root = _resolve_asset(source_root,asset_root)
    package = root/'sensitive_hue'
    if not (package/'model.py').is_file():
        raise FileNotFoundError(f"SensitiveHUE author model absent: {package}")
    name = '_iia_sensitivehue_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        # __init__ eagerly imports the training CLI's top-level `utils` package.
        # Model-only loading does not execute that initializer or edit the source.
        module = types.ModuleType(name)
        module.__path__ = [str(package)]
        module.__package__ = name
        sys.modules[name] = module
    model_module = importlib.import_module(name+'.model')
    model = model_module.SensitiveHUE(step_num_in,f_in,dim_model,head_num,dim_hidden_fc,
                                      encode_layer_num,dropout)
    if not statistical_removal: model.rev_in = _NoRemoval()
    return model


def sensitivehue_loss(reconstruction: Tensor, target: Tensor, log_precision: Tensor, *,
                      alpha: float = 1., probabilistic: bool = True,
                      weighted: bool = True) -> Tensor:
    """Author MTS-NLL; probabilistic=False is the MSE ablation.

    Detached variance weights preserve the author's optimization semantics.
    No clipping is added; overflow must be recorded as a failure, not hidden.
    """
    if reconstruction.shape != target.shape or log_precision.shape != target.shape:
        raise ValueError("all tensors must have equal shapes")
    error = (reconstruction-target).square()
    if not probabilistic: return error.mean()
    nll = error*log_precision.exp()-log_precision
    if weighted:
        variance = (-log_precision).exp().detach()
        return (variance*nll/variance.mean(dim=(0,1)).pow(alpha)).mean()
    return nll.mean()


class FlowMismatchUNet(nn.Module):
    """Appendix C.1 architecture assembled from the pinned Meta FM U-Net.

    Source path points to Meta examples/image/models. Its top-level models.nn
    import is namespaced in memory, with no source overwrite. This source is
    related cited library code, not a Flow Mismatching author release.
    """
    def __init__(self, source_root: str | Path, *, num_classes: int | None = None,
                 model_channels: int = 64, num_res_blocks: int = 3,
                 channel_mult: tuple[int,...] = (1,1,2,2,4,4),
                 attention_resolutions: tuple[int,...] = (4,8,16,32),
                 num_head_channels: int = 32, dropout: float = 0.05,
                 asset_root: str | Path | None = None):
        super().__init__()
        root=_resolve_asset(source_root,asset_root)
        if not (root/'unet.py').is_file() or not (root/'nn.py').is_file():
            raise FileNotFoundError(f"Meta image U-Net source absent: {root}")
        name='_iia_fm_unet_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(root)];package.__package__=name
            sys.modules[name]=package
        if name+'.unet' not in sys.modules:
            module=types.ModuleType(name+'.unet');module.__package__=name
            module.__file__=str(root/'unet.py');sys.modules[module.__name__]=module
            source=(root/'unet.py').read_text(encoding='utf-8')
            source=source.replace('from models.nn import (',f'from {name}.nn import (')
            exec(compile(source,str(root/'unet.py'),'exec'),module.__dict__)
        cls=sys.modules[name+'.unet'].UNetModel
        self.model=cls(in_channels=3,out_channels=3,model_channels=model_channels,
            num_res_blocks=num_res_blocks,channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,num_head_channels=num_head_channels,
            dropout=dropout,num_classes=num_classes,use_scale_shift_norm=True,
            resblock_updown=True,use_new_attention_order=True)
        self.num_classes=num_classes

    def forward(self, images: Tensor, time: Tensor, labels: Tensor | None = None) -> Tensor:
        if self.num_classes is not None and labels is None:
            raise ValueError("class-conditional model requires explicit labels")
        if self.num_classes is None and labels is not None:
            raise ValueError("pooled-category model does not accept labels")
        return self.model(images,time,{'label':labels} if labels is not None else {})
