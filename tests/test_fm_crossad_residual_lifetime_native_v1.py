"""Original full-size layer diagnostics; short tensors are not benchmark data."""
import copy
import importlib
from pathlib import Path
import sys

import pytest
import torch

from iia_benchmark.models.fm_crossad_reference_lifetime_v1 import install_residual_lifetime_policy

RUNTIME=Path('data/public_datasets/flow_matching/crossad_author_runtime_v2').resolve()
sys.path.insert(0,str(RUNTIME))
layers=importlib.import_module('models.CrossAD.EncDec')
attention=importlib.import_module('models.CrossAD.Attention_Blocks')
assert Path(layers.__file__).resolve()==RUNTIME/'models/CrossAD/EncDec.py'


def attn():
    return attention.AttentionLayer(attention.ScaledDotProductAttention(attn_dropout=.1),128,4,proj_dropout=.1)


@pytest.mark.parametrize('kind',['encoder','decoder'])
@pytest.mark.parametrize('training',[False,True])
def test_same_original_arithmetic_full_layer_outputs_gradients_state_and_dropout_rng(kind,training):
    torch.set_num_threads(2);torch.manual_seed(2025)
    if kind=='encoder':
        baseline=layers.EncoderLayer(attn(),128,d_ff=512,norm='layernorm',dropout=.1,activation='gelu')
    else:
        baseline=layers.DecoderLayer(attn(),attn(),128,d_ff=512,norm='layernorm',dropout=.1,activation='gelu')
    changed=copy.deepcopy(baseline);install_residual_lifetime_policy(changed)
    baseline.train(training);changed.train(training)
    first=torch.randn(2,7,128);cross=torch.randn(2,5,128)
    start=torch.get_rng_state().clone();outputs=[];input_grads=[];param_grads=[];end_rng=[]
    for model in (baseline,changed):
        torch.set_rng_state(start)
        x=first.clone().requires_grad_(True);context=cross.clone().requires_grad_(True)
        with torch.set_grad_enabled(training):
            out=model(x) if kind=='encoder' else model(x,context)
            outputs.append([value.detach().clone() for value in out])
            if training:
                sum(value.square().mean() for value in out).backward()
        input_grads.append(None if x.grad is None else x.grad.clone())
        param_grads.append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
        end_rng.append(torch.get_rng_state().clone())
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    assert torch.equal(*end_rng)
    for a,b in zip(*param_grads):assert a is None and b is None or torch.equal(a,b)
    assert input_grads[0] is None and input_grads[1] is None or torch.equal(*input_grads)
    assert baseline.state_dict().keys()==changed.state_dict().keys()
    assert all(torch.equal(value,changed.state_dict()[name]) for name,value in baseline.state_dict().items())
    assert not torch.cuda.is_initialized()
