"""Callable MOMENT/T5 Infini-Channel Mixer integration, without pretraining.

Reuse pinned MOMENT patching, embeddings, normalization, forecasting and
reconstruction heads. Construct the paper's T5-efficient-tiny sized encoder
locally (no automatic Hub downloads). Replace each encoder self-attention with
cross-channel compressive memory that reuses the original Q/K/V/O projections.
Checkpoint equivalence and original pretraining are explicitly unestablished.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
import sys
import types

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fm_supplemental import StaticChannelEmbedding
from .fm_supplemental_author import _resolve_asset


def _moment_components(source_root: str | Path, asset_root: str | Path | None):
    root=_resolve_asset(source_root,asset_root)
    name='_iia_moment_icm_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    for suffix,path in [('',root/'moment'),('.utils',root/'moment/utils'),
                         ('.models',root/'moment/models'),('.models.layers',root/'moment/models/layers')]:
        if name+suffix not in sys.modules:
            module=types.ModuleType(name+suffix);module.__path__=[str(path)];module.__package__=name+suffix
            sys.modules[name+suffix]=module
    # Only these model components are loaded; full MOMENT init eagerly imports
    # data utilities and may automatically request pretrained Hub backbones.
    for suffix in ('.utils.masking','.utils.data','.models.layers.embed','.models.layers.revin'):
        if name+suffix not in sys.modules:
            path=root/'moment'/Path(*suffix.strip('.').split('.')).with_suffix('.py')
            if not path.is_file():raise FileNotFoundError(path)
            source=path.read_text(encoding='utf-8').replace('from moment.',f'from {name}.')
            module=types.ModuleType(name+suffix);module.__package__=(name+suffix).rsplit('.',1)[0]
            module.__file__=str(path);sys.modules[name+suffix]=module
            exec(compile(source,str(path),'exec'),module.__dict__)
    headname=name+'.models.icm_heads'
    if headname not in sys.modules:
        path=root/'moment/models/moment.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('PretrainHead','ForecastingHead')]
        if len(nodes)!=2:raise ValueError('pinned MOMENT heads absent')
        module=types.ModuleType(headname);module.__dict__.update(torch=torch,nn=nn)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
        sys.modules[headname]=module
    return sys.modules[name+'.models.layers.embed'],sys.modules[name+'.models.layers.revin'],sys.modules[headname]


class _T5ICMAttention(nn.Module):
    """Keep T5 local attention including learned relative bias and dropout."""
    def __init__(self, base: nn.Module, train_beta: bool, epsilon: float):
        super().__init__()
        self.base=base
        self.beta=nn.Parameter(torch.zeros(base.n_heads),requires_grad=train_beta)
        self.channels=1
        self.epsilon=epsilon

    def forward(self, hidden_states: Tensor, mask=None, position_bias=None,
                key_value_states=None, output_attentions=False, **kwargs):
        if key_value_states is not None or any(kwargs.get(k) is not None for k in ('past_key_values','past_key_value')):
            raise ValueError('ICM adapter supports uncached encoder self-attention only')
        outputs=self.base(hidden_states,mask=mask,position_bias=position_bias,
                          output_attentions=True,**kwargs)
        weights=outputs[-1]
        bc,n,_=hidden_states.shape;c=self.channels
        if bc%c:raise ValueError('flattened channel batch cannot be grouped')
        b=bc//c;h=self.base.n_heads;d=self.base.key_value_proj_dim
        q,k,v=[layer(hidden_states).reshape(bc,n,h,d).transpose(1,2) for layer in (self.base.q,self.base.k,self.base.v)]
        local=weights@v
        q,k,v=[x.reshape(b,c,h,n,d) for x in (q,k,v)]
        pk,pq=F.elu(k)+1,F.elu(q)+1
        if mask is not None:
            # Encoder masks are additive, 0 for present keys and negative for
            # padding. Padded keys must not enter compressive memory.
            valid=(mask[...,0,:]>=0).reshape(b,c,1,n,1)
            pk=pk*valid
        memory=torch.einsum('bchnd,bchne->bhde',pk,v)
        z=pk.sum((1,3))
        retrieved=torch.einsum('bchnd,bhde->bchne',pq,memory)
        retrieved=retrieved/(torch.einsum('bchnd,bhd->bchn',pq,z).unsqueeze(-1)+self.epsilon)
        gate=self.beta.sigmoid()[None,None,:,None,None]
        blended=gate*retrieved+(1-gate)*local.reshape(b,c,h,n,d)
        blended=blended.reshape(bc,h,n,d).transpose(1,2).reshape(bc,n,h*d)
        result=self.base.o(blended)
        rest=outputs[1:] if output_attentions else outputs[1:-1]
        return (result,)+rest


class MomentICMT5(nn.Module):
    """MOMENT components + every-layer T5 ICM, usable for trainable tasks.

    Inputs [batch,channels,time]; output forecast/reconstruction tensor or
    feature tensor [batch,channels,patches,dimension]. Initialized from scratch;
    this is prepared model code, not the paper's pretrained MOMENT-Tiny weights.
    """
    def __init__(self, source_root: str, *, seq_len: int = 256, patch_len: int = 8,
                 d_model: int = 256, num_layers: int = 4, heads: int = 4,
                 d_ff: int = 1024, forecast_horizon: int = 96,
                 task: str = 'forecast', mixer: str = 'icm', static_channels: int | None = None,
                 train_beta: bool = True, freeze_backbone: bool = False,
                 dropout: float = .1, epsilon: float = 1e-6,
                 asset_root: str | None = None):
        super().__init__()
        if seq_len%patch_len or d_model%heads or task not in ('forecast','reconstruction','features'):
            raise ValueError('invalid patching, head dimension or task')
        if mixer not in ('icm','independent'):raise ValueError('model integration supports ICM or independent channels')
        # Imported on construction so generic module import does not pull HF.
        from transformers import T5Config,T5EncoderModel
        embeds,norms,heads_module=_moment_components(source_root,asset_root)
        self.seq_len,self.patch_len,self.d_model,self.task=seq_len,patch_len,d_model,task
        self.normalizer=norms.RevIN(num_features=1,affine=False)
        self.tokenizer=embeds.Patching(patch_len,patch_len)
        self.patch_embedding=embeds.PatchEmbedding(d_model,seq_len,patch_len,patch_len,dropout=dropout)
        config=T5Config(vocab_size=1,d_model=d_model,d_kv=d_model//heads,d_ff=d_ff,
                        num_layers=num_layers,num_heads=heads,dropout_rate=dropout,
                        feed_forward_proj='relu',use_cache=False,is_encoder_decoder=False,
                        eos_token_id=None,pad_token_id=0,decoder_start_token_id=0)
        self.encoder=T5EncoderModel(config).get_encoder()
        self.icm_layers=nn.ModuleList()
        if mixer=='icm':
            for block in self.encoder.block:
                attention=_T5ICMAttention(block.layer[0].SelfAttention,train_beta,epsilon)
                block.layer[0].SelfAttention=attention;self.icm_layers.append(attention)
        self.static=StaticChannelEmbedding(static_channels,d_model) if static_channels is not None else None
        self.head=(heads_module.ForecastingHead((seq_len//patch_len)*d_model,forecast_horizon)
                   if task=='forecast' else heads_module.PretrainHead(d_model,patch_len,head_dropout=dropout))
        if freeze_backbone:
            for module in (self.encoder,self.patch_embedding):
                for p in module.parameters():p.requires_grad_(False)
            for layer in self.icm_layers:layer.beta.requires_grad_(train_beta)

    def forward(self, values: Tensor, input_mask: Tensor | None = None,
                reconstruction_mask: Tensor | None = None) -> Tensor:
        if values.ndim!=3 or values.shape[-1]!=self.seq_len:
            raise ValueError('expected [batch,channels,configured sequence length]')
        b,c,_=values.shape
        if input_mask is None:input_mask=values.new_ones(b,self.seq_len)
        if reconstruction_mask is None:reconstruction_mask=input_mask
        if input_mask.shape!=(b,self.seq_len) or reconstruction_mask.shape!=input_mask.shape:
            raise ValueError('masks must be [batch,time]')
        normalized=self.normalizer(values,mask=input_mask*reconstruction_mask,mode='norm')
        tokens=self.patch_embedding(self.tokenizer(torch.nan_to_num(normalized)),mask=reconstruction_mask)
        if self.static is not None:tokens=self.static(tokens)
        n=tokens.shape[2]
        valid=input_mask.unfold(-1,self.patch_len,self.patch_len).sum(-1).eq(self.patch_len)
        for layer in self.icm_layers:layer.channels=c
        try:
            output=self.encoder(inputs_embeds=tokens.reshape(b*c,n,self.d_model),
                                attention_mask=valid.repeat_interleave(c,0)).last_hidden_state
        finally:
            for layer in self.icm_layers:layer.channels=1
        output=output.reshape(b,c,n,self.d_model)
        if self.task=='features':return output
        predicted=self.head(output)
        return self.normalizer(predicted,mode='denorm')
