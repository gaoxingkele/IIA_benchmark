"""Source-fidelity and lifecycle checks; synthetic inputs are plumbing only."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from scripts.flow_matching import run_tranad_author_all_entities_v1 as s
from iia_benchmark.models import tranad_author_source_v1 as runtime


def config(variant='TranAD'):
    return s.read(s.ROOT/(s.MODEL_PREFIX+variant+'.json'))


@pytest.mark.parametrize('variant', runtime.VARIANTS)
def test_every_exact_source_branch_has_finite_forward_backward_and_source_parameterization(variant):
    torch.set_num_threads(2); torch.manual_seed(51)
    model, ns = runtime.author_model(config(variant), 'SMAP', variant, 3)
    assert model.lr == .001 and model.batch == 128 and model.n_window == 10
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    model.eval(); x = torch.arange(60, dtype=torch.float64).reshape(10,2,3)/100
    y = model(x, x[-1:].clone()); outputs = y if isinstance(y,tuple) else (y,)
    assert all(v.shape == (1,2,3) and torch.isfinite(v).all() for v in outputs)
    sum(v.square().mean() for v in outputs).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if variant != 'TranAD_Transformer':
        layers = [m for m in model.modules() if isinstance(m, ns['TransformerEncoderLayer'])]
        assert len(layers) == 1 and layers[0].activation.negative_slope is True


def test_source_position_encoding_is_sin_plus_cos_for_every_dimension():
    model, ns = runtime.author_model(config(), 'SMD', 'TranAD', 3)
    pe = model.pos_encoder.pe
    phase = torch.arange(10).float().unsqueeze(1)*torch.exp(torch.arange(6).float()*(-np.log(10000.)/6))
    assert torch.equal(pe[:,0], (torch.sin(phase)+torch.cos(phase)).double())
    assert torch.equal(pe[0], torch.ones_like(pe[0]))
    assert ns['lm'] == (.99995, 1.06) and model.lr == .0001


def test_source_windows_preserve_previous_timestamp_padding_and_every_tail():
    model, ns = runtime.author_model(config(), 'MSL', 'TranAD', 2)
    x = torch.arange(26,dtype=torch.float64).reshape(13,2)
    w = ns['convert_to_windows'](x,model)
    assert w.shape == (13,10,2)
    assert torch.equal(w[0],x[0].repeat(10,1))
    assert torch.equal(w[10],x[:10]) and torch.equal(w[-1],x[2:12])
    assert not torch.equal(w[-1,-1],x[-1])


def test_source_functions_and_custom_layer_AST_are_executed_without_rewrites():
    c=config();ns=runtime.load_author_runtime(c,'SMAP');root=s.ROOT/c['source_root']
    for name,path in [('backprop','main.py'),('convert_to_windows','main.py'),('TranAD','src/models.py'),('PositionalEncoding','src/dlutils.py')]:
        assert str(root/path).replace('\\','/').endswith(str(Path(ns[name].__code__.co_filename if hasattr(ns[name],'__code__') else ns[name].forward.__code__.co_filename)).replace('\\','/'))
    tree=runtime.definition(root/'main.py','backprop')
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="'TranAD' in model.name")
    loaders=[n for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DataLoader']
    assert len(loaders)==1 and not any(k.arg=='shuffle' for k in loaders[0].keywords)
    assert 'len(data)' in ast.unparse(branch) and 'torch.DoubleTensor(data)' in ast.unparse(branch)


def test_sequential_container_matches_direct_original_layer_call_and_state_keys():
    torch.manual_seed(12);model,ns=runtime.author_model(config(),'SMAP','TranAD',3)
    model.eval();x=torch.randn(10,2,6,dtype=torch.float64)
    expected=model.transformer_encoder.layers[0](x)
    assert torch.equal(model.transformer_encoder(x),expected)
    keys=model.state_dict()
    assert 'transformer_encoder.layers.0.self_attn.in_proj_weight' in keys
    assert not any('.norm' in name for name in keys)


def test_changed_original_source_receipt_or_unknown_branch_rejected():
    c=deepcopy(config());c['source_receipts'][0]['sha256']='0'*64
    with pytest.raises(ValueError):runtime.load_author_runtime(c,'SMAP')
    with pytest.raises(ValueError):runtime.author_model(config(),'SMAP','unknown',3)
    with pytest.raises(ValueError):runtime.load_author_runtime(config(),'PSM')


def test_all109_entities_five_methods_six_seeds_and_primary_cohorts_not_narrowed():
    q=s.registration()
    assert len(q['entities'])==109 and len(q['jobs'])==3270
    assert q['required_primary_five_seed_cohorts']==545 and q['epochs']==5
    assert sum(j['original_CLI_default_entity'] for j in q['jobs'])==90
    assert sum(j['seed']==1 for j in q['jobs'])==545
    assert len({j['id'] for j in q['jobs']})==3270
    assert set(j['variant'] for j in q['jobs'])==set(runtime.VARIANTS)
    assert q['settings']['minimum_commit_headroom_bytes']==40*2**30
    assert q['resource_lock']==s.read(s.ROOT/q['resource_reference_configuration'])['resource_lock']
    readonly=s.read(s.ROOT/q['preparation_resource_reference_configuration'])
    assert q['preparation_resource_lock']==readonly['resource_lock'] and q['preparation_settings']==readonly['settings']
    q['jobs'].pop()
    with pytest.raises(ValueError):s.verify(q)


def test_metadata_entry_does_not_import_numeric_frameworks():
    code='from scripts.flow_matching import run_tranad_author_all_entities_v1 as s;import sys;s.registration();assert "torch" not in sys.modules and "numpy" not in sys.modules'
    subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,check=True)


@pytest.mark.parametrize('variant', runtime.VARIANTS)
def test_original_five_epoch_training_optimizer_schedule_and_checkpoint_roundtrip(variant):
    # A short synthetic lifecycle is not an original-data benchmark condition.
    torch.set_num_threads(2);torch.manual_seed(76)
    c=config(variant);model,ns=runtime.author_model(c,'SMAP',variant,3)
    data=torch.arange(39,dtype=torch.float64).reshape(13,3)/100
    windows=ns['convert_to_windows'](data,model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=model.lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5,.9)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    for epoch in range(5):
        loss,lr=ns['backprop'](epoch,model,windows,data,optimizer,scheduler)
        assert np.isfinite(loss)
    assert lr==pytest.approx(.0009)
    assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    assert all(int(v['step'])==5 for v in optimizer.state.values())
    model.eval();a=ns['backprop'](0,model,windows,data,optimizer,scheduler,training=False)
    restored,second=runtime.author_model(c,'SMAP',variant,3);restored.load_state_dict(model.state_dict());restored.eval()
    b=second['backprop'](0,restored,windows,data,optimizer,scheduler,training=False)
    assert all(x.dtype==y.dtype and x.shape==y.shape==(13,3) and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def test_undefined_author_metric_is_retained_without_zero_filling():
    assert s.scalar_outcomes(dict(f1=.3,roc=float('nan'))) == dict(values=dict(f1=.3,roc=None),nonfinite_fields=['roc'])


def test_metric_worker_cannot_accept_an_unguarded_source_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):s.prior_stage(dict(artifact_root=str(tmp_path),state_root='state'),dict(id='case'),'replay')
