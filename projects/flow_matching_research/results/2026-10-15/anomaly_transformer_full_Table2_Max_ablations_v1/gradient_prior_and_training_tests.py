"""Joint-Max gradients, exact fixed Gaussian and full-source training checks."""
from copy import deepcopy
import inspect
import json
import math
import os
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_paper_max_ablation_v1 as subject


def config_fixture(monkeypatch):
    original=subject.read(subject.ROOT/subject.PARENT)
    monkeypatch.setattr(subject.parent,'verify',lambda _:subject.read(subject.ROOT/subject.parent.CONTRACT))
    config=dict(original,parent_registration=subject.PARENT,parent_registration_sha256=subject.sha(subject.ROOT/subject.PARENT),
        base_training=original['training'],variants=subject.VARIANTS,primary_score_stream='all.multiplication',
        training=dict(original['training'],backward_strategy='one_joint_direct_Max_backward_then_one_Adam_step',
            loss_reduction='per_window_Frobenius_squared_minus_lambda_L1_joint_discrepancy_then_batch_mean'),
        optimizer_update='one_joint_direct_Max_backward_then_one_Adam_step',fixed_prior_scale=1,
        paired_shared_parameter_initialization=True,source_receipts=[])
    config['jobs']=[dict(id=f"anomaly_transformer__paper_ablation__{v}__{j['dataset'].lower()}__seed{j['seed']}",
                        dataset=j['dataset'],seed=j['seed'],variant=v) for v in subject.VARIANTS for j in original['jobs']]
    return config


def test_both_complete_training_ablations_are_present(monkeypatch):
    config=config_fixture(monkeypatch);subject.verify(config)
    assert len(config['jobs'])==50 and len({j['id'] for j in config['jobs']})==50


@pytest.mark.parametrize('mutation',['variant','job','epochs','guard','sigma','backward','primary','paired','actual_training'])
def test_rejects_shortened_or_changed_ablation_protocol(monkeypatch,mutation):
    config=deepcopy(config_fixture(monkeypatch))
    if mutation=='variant':config['variants'].pop()
    elif mutation=='job':config['jobs'].pop()
    elif mutation=='epochs':config['base_training']['maximum_epochs']=3
    elif mutation=='guard':config['settings']['minimum_commit_headroom_bytes']=32*1024**3
    elif mutation=='sigma':config['fixed_prior_scale']=2
    elif mutation=='backward':config['optimizer_update']='two_detached_minimax_backwards'
    elif mutation=='primary':config['primary_score_stream']='best_test_stream'
    elif mutation=='paired':config['paired_shared_parameter_initialization']=False
    else:config['training']['backward_strategy']='two_detached_minimax_backwards'
    with pytest.raises(ValueError):subject.verify(config)


def test_direct_Max_uses_joint_prior_and_series_gradients_and_equation4_loss():
    import torch
    p=torch.tensor([[[[.8,.2],[.8,.2]]]],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([[[[.4,.6],[.4,.6]]]],dtype=torch.float64,requires_grad=True)
    target=torch.ones(1,2,2,dtype=torch.float64);reconstruction=torch.zeros_like(target,requires_grad=True)
    minimum,maximum,rec=subject.direct_max_losses(reconstruction,target,[p],[s])
    expected_discrepancy=sum((a-b)*(math.log(a+.0001)-math.log(b+.0001)) for a,b in zip([.8,.2],[.4,.6]))
    assert math.isclose(float(maximum.detach()),4-3*2*expected_discrepancy,abs_tol=1e-12)
    assert minimum is maximum and float(rec.detach())==4
    gp,gs=torch.autograd.grad(maximum,[p,s])
    assert torch.any(gp!=0) and torch.any(gs!=0)


def test_backward_probe_counts_autograd_execution_once_under_CPU_device_context(monkeypatch):
    import torch
    calls=[];original=torch.autograd.backward
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'backward',counted)
    torch.set_default_device('cpu')
    value=torch.ones(2,requires_grad=True);value.square().sum().backward()
    assert len(calls)==1 and torch.equal(value.grad,torch.full_like(value,2))


def test_fixed_sigma1_removes_projection_preserves_shared_initialization_and_exact_prior():
    import torch
    source=subject.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original/model/attn.py'
    raw=source.read_text(encoding='utf-8');modules=[]
    for variant in subject.VARIANTS:
        namespace={};exec(compile(subject.patched_attention('model/attn.py',raw,variant),str(source),'exec'),namespace)
        torch.manual_seed(1103)
        model=namespace['AttentionLayer'](namespace['AnomalyAttention'](100,False,output_attention=True),32,4)
        modules.append(model)
    learned,fixed=modules
    learned_state,fixed_state=learned.state_dict(),fixed.state_dict()
    assert all(torch.equal(v,learned_state[k]) for k,v in fixed_state.items())
    assert not any('sigma_projection' in k for k in fixed_state)
    assert sum(p.numel() for p in learned.parameters())-sum(p.numel() for p in fixed.parameters())==32*4+4
    x=torch.randn(2,100,32)
    learned_out=learned(x,x,x,None);fixed_out=fixed(x,x,x,None)
    assert torch.equal(learned_out[0],fixed_out[0])
    assert torch.equal(fixed_out[3],torch.ones_like(fixed_out[3]))
    expected=1/math.sqrt(2*math.pi)
    assert math.isclose(float(fixed_out[2][0,0,50,50]),expected,abs_tol=1e-7)
    assert math.isclose(float(fixed_out[2][0,0,50,51]),expected*math.exp(-.5),abs_tol=1e-7)


@pytest.mark.parametrize('variant',subject.VARIANTS)
def test_actual_full_width_training_uses_one_backward_per_update(variant,tmp_path):
    code=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import run_anomaly_transformer_paper_max_ablation_v1 as s
root=pathlib.Path(sys.argv[1]);variant=sys.argv[2];raw=root/'data/toy';raw.mkdir(parents=True)
configs=root/'configs/datasets';configs.mkdir(parents=True)
rng=np.random.default_rng(1103)
np.save(raw/'train.npy',rng.normal(size=(500,3)));np.save(raw/'test.npy',rng.normal(size=(100,3)))
labels=np.zeros(100,dtype=np.int8);labels[20:40]=1;np.save(raw/'labels.npy',labels)
(configs/'mtsad_smd.json').write_text(json.dumps(dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='labels.npy'))))
contract=dict(reported_table13_splits=dict(SMD=dict(train=400,validation=100,test=100,features=3)),paper_evaluation=dict(r_percent=dict(SMD=.5),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
source=s.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
s.parent.ROOT=root
config=dict(artifact_root=str(root/'artifacts'),data_cache_root=str(root/'cache'),data_root='data',source_directory=str(source),contract_sha256='fixture',explicit_reconstruction_choices=['fixture only'],boundary='fixture only, not benchmark performance')
path=root/'config.json';path.write_text(json.dumps(config));job=dict(id='fixture',dataset='SMD',seed=1103,variant=variant)
backwards=[];original=torch.autograd.backward
def counted(*args,**kwargs):backwards.append(1);return original(*args,**kwargs)
torch.autograd.backward=counted
s.bound_worker(variant,verify_function=lambda _:contract,root=root)(config,job,path)
r=json.loads((root/'artifacts/fixture/result.json').read_text())
assert r['parameter_count']>4000000 and 4<=len(r['epochs'])<=10
assert r['actual_backward_calls']==r['optimizer_updates']==len(backwards)==len(r['epochs'])
assert len(r['metrics'])==7 and not r['checkpoint_selection_uses_test'] and not r['threshold_calibration_uses_test']
assert all(e['train_losses'][0]==e['train_losses'][1] for e in r['epochs'])
weights=torch.load(root/'artifacts/fixture/model.pt',map_location='cpu',weights_only=True)
assert any('sigma_projection' in k for k in weights)==(variant=='learnable_prior_max')
with np.load(root/'artifacts/fixture/test_scores.npz') as a:assert a['component.layer_discrepancies'].shape==(100,3)
'''
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(tmp_path),variant],cwd=subject.ROOT,env=environment,
                     capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout[-2000:]+p.stderr[-4000:]


def test_actual_resource_interface_is_cold():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1;from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
