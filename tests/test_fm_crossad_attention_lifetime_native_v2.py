"""Real source attention/layers: short diagnostics are not PSM performance."""
import copy
import importlib
from pathlib import Path
import sys
import weakref
import pytest
import torch
from iia_benchmark.models.fm_crossad_attention_lifetime_v2 import install_attention_lifetime_policy
from iia_benchmark.models.fm_crossad_reference_lifetime_v1 import install_residual_lifetime_policy

RUNTIME=Path('data/public_datasets/flow_matching/crossad_author_runtime_v2').resolve()
sys.path.insert(0,str(RUNTIME))
attention=importlib.import_module('models.CrossAD.Attention_Blocks')
layers=importlib.import_module('models.CrossAD.EncDec')
assert Path(attention.__file__).resolve()==RUNTIME/'models/CrossAD/Attention_Blocks.py'


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('mask_kind',['none','boolean','additive'])
def test_source_attention_output_qkv_gradients_masks_and_dropout_rng_exact(training,mask_kind):
    torch.set_num_threads(2);torch.manual_seed(2025)
    baseline=attention.ScaledDotProductAttention(attn_dropout=.1)
    changed=copy.deepcopy(baseline);install_attention_lifetime_policy(changed)
    baseline.train(training);changed.train(training)
    q,k,v=torch.randn(2,4,7,32),torch.randn(2,4,32,5),torch.randn(2,4,5,32)
    mask=None
    if mask_kind=='boolean':mask=torch.zeros(7,5,dtype=torch.bool);mask[:,0]=True
    if mask_kind=='additive':mask=torch.randn(7,5)*.1
    initial=torch.get_rng_state().clone();outputs=[];gradients=[];rng=[]
    for model in (baseline,changed):
        torch.set_rng_state(initial)
        args=[x.clone().requires_grad_(training) for x in (q,k,v)]
        with torch.set_grad_enabled(training):
            result=model(*args,attn_mask=mask);outputs.append([x.detach().clone() for x in result])
            if training:sum(x.square().mean() for x in result).backward()
        gradients.append([None if x.grad is None else x.grad.clone() for x in args])
        rng.append(torch.get_rng_state().clone())
    assert all(torch.equal(a,b) for a,b in zip(*outputs)) and torch.equal(*rng)
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(*gradients))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind',['encoder','decoder'])
@pytest.mark.parametrize('training',[False,True])
def test_combined_policy_preserves_every_layer_output_parameter_gradient_and_rng(kind,training):
    torch.set_num_threads(2);torch.manual_seed(2025)
    def attn():return attention.AttentionLayer(attention.ScaledDotProductAttention(attn_dropout=.1),128,4,proj_dropout=.1)
    baseline=(layers.EncoderLayer(attn(),128,d_ff=512,norm='layernorm',dropout=.1,activation='gelu') if kind=='encoder'
        else layers.DecoderLayer(attn(),attn(),128,d_ff=512,norm='layernorm',dropout=.1,activation='gelu'))
    changed=copy.deepcopy(baseline);install_residual_lifetime_policy(changed);install_attention_lifetime_policy(changed)
    baseline.train(training);changed.train(training)
    first,context=torch.randn(2,7,128),torch.randn(2,5,128)
    initial=torch.get_rng_state().clone();outputs=[];gradients=[];rng=[]
    for model in (baseline,changed):
        torch.set_rng_state(initial);x=first.clone().requires_grad_(training);cross=context.clone().requires_grad_(training)
        with torch.set_grad_enabled(training):
            result=model(x) if kind=='encoder' else model(x,cross)
            outputs.append([y.detach().clone() for y in result])
            if training:sum(y.square().mean() for y in result).backward()
        gradients.append([None if p.grad is None else p.grad.clone() for p in [x,cross,*model.parameters()]])
        rng.append(torch.get_rng_state().clone())
    assert all(torch.equal(a,b) for a,b in zip(*outputs)) and torch.equal(*rng)
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(*gradients))
    assert all(torch.equal(value,changed.state_dict()[name]) for name,value in baseline.state_dict().items())
    assert not torch.cuda.is_initialized()


def test_dead_score_storage_is_released_before_original_dropout_without_changing_math(monkeypatch):
    model=attention.ScaledDotProductAttention(attn_dropout=.1).eval();install_attention_lifetime_policy(model)
    refs=[];softmax=attention.F.softmax
    def capture(scores,*args,**kwargs):
        refs.append(weakref.ref(scores));return softmax(scores,*args,**kwargs)
    monkeypatch.setattr(attention.F,'softmax',capture)
    def before_dropout(module,args):assert refs[-1]() is None
    handle=model.attn_dropout.register_forward_pre_hook(before_dropout)
    with torch.no_grad():model(torch.randn(2,4,7,32),torch.randn(2,4,32,5),torch.randn(2,4,5,32))
    handle.remove();assert not torch.cuda.is_initialized()
