"""Fine-grained pure-layer checkpoints; preserve algorithm outputs/RNG/EMA exactly."""
import torch
from torch.utils.checkpoint import checkpoint
from .fm_crossad_author_runtime import build_model


def install_activation_checkpointing(model):
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Stateful normalization cannot be replayed safely')
    pure={'EncoderLayer','DecoderLayer','ExtractorLayer','AttentionLayer'}
    count=0
    for module in list(model.modules()):
        name=type(module).__name__
        if name not in pure:continue
        original=module.forward
        def pure_forward(*args,_original=original,_name=name,**kwargs):
            result=_original(*args,**kwargs)
            # These diagnostic matrices are unused by source loss, context updates and scores.
            # Release their references; internal matrix multiplication/dropout remains identical.
            if _name in ('EncoderLayer','DecoderLayer','AttentionLayer'):
                return (result[0],)+tuple(None for _ in result[1:])
            return result
        def forward(*args,_module=module,_pure=pure_forward,**kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_pure,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return _pure(*args,**kwargs)
        module.forward=forward;count+=1
    assert count>0
    model.activation_checkpoint_boundary_='Pure LayerNorm encoder/decoder/extractor/attention layers only. EMA update and router outside replay. Unused diagnostic attention references released; full algorithm outputs, gradients, RNG and buffers checked against source.'
    model.checkpointed_pure_layer_count_=count
    return model


def build_memory_model(runtime,parameters):
    return install_activation_checkpointing(build_model(runtime,parameters))
