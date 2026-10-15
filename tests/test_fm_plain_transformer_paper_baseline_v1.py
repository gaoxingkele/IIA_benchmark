"""Plain attention identity, removed prior graph and full-budget integration."""
from copy import deepcopy
import json
import math
import os
import subprocess
import sys

import pytest

from scripts.flow_matching import run_plain_transformer_paper_baseline_v1 as subject


def config_fixture(monkeypatch):
    original=subject.read(subject.ROOT/subject.PARENT)
    monkeypatch.setattr(subject.parent,'verify',lambda _:subject.read(subject.ROOT/subject.parent.CONTRACT))
    config=dict(original,parent_registration=subject.PARENT,parent_registration_sha256=subject.sha(subject.ROOT/subject.PARENT),
        base_training=original['training'],training=dict({k:v for k,v in original['training'].items() if k not in ['discrepancy_weight','kl_log_epsilon']},
            backward_strategy='one_reconstruction_backward_then_one_Adam_step',
            loss_reduction='per_window_Frobenius_squared_then_batch_mean'),
        primary_score_stream='all.reconstruction',score_streams=subject.STREAMS,prior_path_present=False,
        paired_shared_parameter_initialization=True,source_receipts=[])
    config['jobs']=[dict(id=f"plain_transformer__paper_Table2__{j['dataset'].lower()}__seed{j['seed']}",
                        dataset=j['dataset'],seed=j['seed'],variant='plain_attention_reconstruction') for j in original['jobs']]
    return config


def test_all_plain_baseline_dataset_seed_slots_present(monkeypatch):
    c=config_fixture(monkeypatch);subject.verify(c);assert len(c['jobs'])==25


@pytest.mark.parametrize('mutation',['jobs','width','epochs','guard','prior','criterion','backward','paired'])
def test_rejects_narrowed_or_different_baseline(monkeypatch,mutation):
    c=deepcopy(config_fixture(monkeypatch))
    if mutation=='jobs':c['jobs'].pop()
    elif mutation=='width':c['model']['d_model']=32
    elif mutation=='epochs':c['training']['maximum_epochs']=3
    elif mutation=='guard':c['settings']['minimum_commit_headroom_bytes']=32*1024**3
    elif mutation=='prior':c['prior_path_present']=True
    elif mutation=='criterion':c['primary_score_stream']='all.multiplication'
    elif mutation=='backward':c['training']['backward_strategy']='minimax'
    else:c['paired_shared_parameter_initialization']=False
    with pytest.raises(ValueError):subject.verify(c)


def test_plain_attention_matches_scaled_dot_product_and_shared_initial_weights():
    import torch
    raw=(subject.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original/model/attn.py').read_text(encoding='utf-8')
    modules=[]
    for source in [subject.parent.author.patched('model/attn.py',raw),subject.patched_attention('model/attn.py',raw)]:
        namespace={};exec(compile(source,'source_attention','exec'),namespace)
        torch.manual_seed(1103)
        modules.append(namespace['AttentionLayer'](namespace['AnomalyAttention'](100,False,output_attention=True),32,4))
    original,plain=modules
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in plain.state_dict().items())
    assert not hasattr(plain.inner_attention,'distances')
    assert not any('sigma_projection' in k for k in plain.state_dict())
    x=torch.randn(2,100,32);out,series,prior,sigma=plain(x,x,x,None)
    assert prior is None and sigma is None
    q=plain.query_projection(x).view(2,100,4,8);k=plain.key_projection(x).view(2,100,4,8)
    v=plain.value_projection(x).view(2,100,4,8)
    expected_series=torch.softmax((1/math.sqrt(8))*torch.einsum('blhe,bshe->bhls',q,k),dim=-1)
    expected=plain.out_projection(torch.einsum('bhls,bshd->blhd',expected_series,v).contiguous().view(2,100,32))
    assert torch.equal(series,expected_series) and torch.equal(out,expected)
    original_out=original(x,x,x,None)
    assert torch.equal(out,original_out[0]) and torch.equal(series,original_out[1])


def test_reconstruction_loss_and_entropy_diagnostic_never_use_prior_scores():
    import torch
    x=torch.ones(1,2,2);y=torch.zeros_like(x,requires_grad=True)
    series=[torch.full((1,2,2,2),.5)]*3
    losses=subject.reconstruction_losses(y,x,[None]*3,series)
    assert losses[0] is losses[1] and float(losses[0].detach())==4
    scores,rec,entropy=subject.reconstruction_scores(y,x,[None]*3,series)
    assert list(scores)==['all.reconstruction'] and torch.equal(rec,torch.ones_like(rec))
    assert torch.allclose(entropy,torch.full_like(entropy,math.log(2)))
    with pytest.raises(ValueError):subject.reconstruction_losses(y,x,[torch.ones(1)],series)


def test_full_width_training_without_prior_and_complete_reconstruction_scores(tmp_path):
    code=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import run_plain_transformer_paper_baseline_v1 as s
root=pathlib.Path(sys.argv[1]);raw=root/'data/toy';raw.mkdir(parents=True)
configs=root/'configs/datasets';configs.mkdir(parents=True)
rng=np.random.default_rng(1103)
np.save(raw/'train.npy',rng.normal(size=(500,3)));np.save(raw/'test.npy',rng.normal(size=(100,3)))
labels=np.zeros(100,dtype=np.int8);labels[20:40]=1;np.save(raw/'labels.npy',labels)
(configs/'mtsad_smd.json').write_text(json.dumps(dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='labels.npy'))))
contract=dict(reported_table13_splits=dict(SMD=dict(train=400,validation=100,test=100,features=3)),paper_evaluation=dict(r_percent=dict(SMD=.5),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
source=s.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original';s.parent.ROOT=root
config=dict(artifact_root=str(root/'artifacts'),data_cache_root=str(root/'cache'),data_root='data',source_directory=str(source),contract_sha256='fixture',explicit_reconstruction_choices=['fixture only'],boundary='fixture only, not benchmark performance')
path=root/'config.json';path.write_text(json.dumps(config));job=dict(id='fixture',dataset='SMD',seed=1103)
calls=[];original=torch.autograd.backward
def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
torch.autograd.backward=counted
s.bound_worker(verify_function=lambda _:contract,root=root)(config,job,path)
r=json.loads((root/'artifacts/fixture/result.json').read_text())
assert r['parameter_count']>4000000 and 4<=len(r['epochs'])<=10
assert r['actual_backward_calls']==r['optimizer_updates']==len(calls)==len(r['epochs'])
assert r['covered_points']==dict(train=400,validation=100,test=100)
assert list(r['metrics'])==['all.reconstruction'] and not r['checkpoint_selection_uses_test']
assert r['metrics']['all.reconstruction']['fixed_delta_is_diagnostic_only_for_plain_Transformer']
weights=torch.load(root/'artifacts/fixture/model.pt',map_location='cpu',weights_only=True)
assert not any('sigma_projection' in k for k in weights)
for split,n in [('train',400),('validation',100),('test',100)]:
 with np.load(root/'artifacts/fixture'/(split+'_scores.npz')) as a:
  assert a['all.reconstruction'].shape==(n,)
  assert a['diagnostic.layer_attention_entropy'].shape==(n,3)
  assert 'component.layer_discrepancies' not in a.files and 'all.multiplication' not in a.files
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(tmp_path)],cwd=subject.ROOT,env=env,
                     capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout[-2000:]+p.stderr[-4000:]


def test_real_resource_interface_stays_numerically_cold():
    code='import sys;import scripts.flow_matching.run_plain_transformer_paper_baseline_v1;from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
