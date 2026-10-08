"""Recompute pure encoder/decoder activations; never checkpoint EMA bank updates."""
import torch
from torch.utils.checkpoint import checkpoint
from .fm_crossad_author_runtime import build_model


def install_activation_checkpointing(model):
    for name in ['encoder', 'decoder']:
        module = getattr(model, name)
        original = module.forward
        def forward(*args, _module=module, _original=original, **kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)
        module.forward = forward
    model.activation_checkpoint_boundary_ = 'Only pure encoder/decoder recomputed; source EMA/router/context runs once; dropout RNG preserved. Full batch and source optimizer/early-stopping unchanged.'
    return model


def build_memory_model(runtime, model_parameters):
    return install_activation_checkpointing(build_model(runtime, model_parameters))
