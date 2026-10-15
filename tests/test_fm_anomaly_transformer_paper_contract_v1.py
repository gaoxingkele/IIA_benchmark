"""Analytical, leakage and full-source integration checks, not performance runs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as subject


def configuration():
    path=subject.ROOT/subject.CONTRACT
    return dict(contract=subject.CONTRACT,contract_sha256=subject.sha(path),
        model=dict(d_model=512,d_ff=512,n_heads=8,e_layers=3,dropout=0.0,window=100),
        training=dict(batch_size=32,maximum_epochs=10,window_stride=100,learning_rate=.0001,
            discrepancy_weight=3,patience=3,kl_log_epsilon=.0001,cpu_threads=2,optimizer='Adam',
            backward_strategy='two_detached_losses_one_accumulated_Adam_update',
            loss_reduction='per_window_Frobenius_squared_and_L1_discrepancy_then_batch_mean',
            learning_rate_schedule='inherited_author_halving_after_epoch_two'),
        seeds=list(range(1103,1108)),score_streams=subject.STREAMS,
        original_scope=subject.read(path)['original_scope'],source_receipts=[],
        settings=dict(minimum_available_memory_bytes=32*1024**3,minimum_commit_headroom_bytes=40*1024**3,
                      emergency_commit_headroom_bytes=16*1024**3),
        jobs=[dict(id=f'anomaly_transformer__paper_contract_reconstruction__{d.lower()}__seed{s}',dataset=d,seed=s)
              for d in ['SMD','MSL','SMAP','PSM','SWaT'] for s in range(1103,1108)])


def test_complete_paper_reconstruction_contract():
    assert subject.verify(configuration())['reported_table13_splits']['SWaT']['train']==396000


@pytest.mark.parametrize('mutation',['width','batch','stride','epochs','temperature','matrix','seed','streams','scope','guard'])
def test_rejects_incomplete_or_changed_scientific_contract(mutation):
    c=deepcopy(configuration())
    if mutation=='width':c['model']['d_model']=32
    elif mutation=='batch':c['training']['batch_size']=256
    elif mutation=='stride':c['training']['window_stride']=1
    elif mutation=='epochs':c['training']['maximum_epochs']=3
    elif mutation=='temperature':c['training']['temperature']=50
    elif mutation=='matrix':c['jobs'].pop()
    elif mutation=='seed':c['jobs'][0]['seed']=42
    elif mutation=='streams':c['score_streams'].pop()
    elif mutation=='scope':c['original_scope']={}
    else:c['settings']['minimum_commit_headroom_bytes']=32*1024**3
    with pytest.raises(ValueError):subject.verify(c)


def test_KL_of_head_means_matches_analytical_zero_not_average_KL():
    import torch
    p=torch.tensor([[[[.9,.1],[.9,.1]],[[.1,.9],[.1,.9]]]],dtype=torch.float64)
    s=torch.full_like(p,.5)
    value,_=subject.head_mean_discrepancy([p],[s])
    assert torch.equal(value,torch.zeros_like(value))
    source_order=(p*(torch.log(p+.0001)-torch.log(s+.0001))+
                  s*(torch.log(s+.0001)-torch.log(p+.0001))).sum(-1).mean(1)
    assert torch.all(source_order>.8)


def test_minimax_stops_only_the_specified_distribution_gradient():
    import torch
    p=torch.tensor([[[[.8,.2],[.7,.3]]]],requires_grad=True)
    s=torch.tensor([[[[.4,.6],[.5,.5]]]],requires_grad=True)
    x=torch.ones(1,2,2);y=torch.zeros_like(x,requires_grad=True)
    minimum,maximum,_=subject.minimax_losses(y,x,[p],[s])
    gp,gs=torch.autograd.grad(minimum,[p,s],allow_unused=True,retain_graph=True)
    assert gp is not None and torch.any(gp!=0) and gs is None
    gp,gs=torch.autograd.grad(maximum,[p,s],allow_unused=True)
    assert gp is None and gs is not None and torch.any(gs!=0)


def test_equation5_uses_full_Frobenius_and_both_reconstruction_phases():
    import torch
    x=torch.tensor([[[1.,2.],[3.,4.]]]);y=torch.zeros_like(x,requires_grad=True)
    p=torch.full((1,1,2,2),.5)
    minimum,maximum,rec=subject.minimax_losses(y,x,[p],[p])
    assert float(rec.detach())==30 and float(minimum.detach())==30 and float(maximum.detach())==30
    minimum.backward(retain_graph=True);maximum.backward()
    assert torch.equal(y.grad,-4*x)


def test_layer_averaging_and_paper_criteria_are_temperature_free():
    import torch
    p=torch.tensor([[[[.8,.2],[.5,.5]]]])
    s=torch.tensor([[[[.3,.7],[.5,.5]]]])
    d1,_=subject.head_mean_discrepancy([p],[s]);d3,_=subject.head_mean_discrepancy([p,p,p],[s,s,s])
    assert torch.allclose(d1,d3)
    x=torch.zeros(1,2,2);y=torch.ones_like(x)
    scores,rec,layers=subject.score_components(y,x,[p,p,p],[s,s,s])
    assert set(scores)==set(subject.STREAMS)
    assert torch.allclose(scores['all.association'],torch.softmax(-d1,dim=-1))
    assert not torch.allclose(scores['all.association'],torch.softmax(-50*3*d1,dim=-1))
    assert torch.equal(rec,torch.ones_like(rec))


def test_threshold_uses_independent_validation_and_PA_is_separate():
    labels=np.array([0,1,1,1,0]);scores=np.array([0.,0.,0.,10.,0.]);validation=np.arange(5.)
    result=subject.evaluation(labels,scores,validation,20,.1,[.5,1,1.5,2,10,20,30])
    ratio=result['validation_ratio']
    assert ratio['threshold']==3.2
    assert ratio['no_PA']['f1']==.5 and ratio['paper_PA']['f1']==1
    changed=subject.evaluation(labels,scores*100,validation,20,.1,[.5,1,1.5,2,10,20,30])
    assert changed['validation_ratio']['threshold']==ratio['threshold']
    assert not result['test_features_or_labels_used_for_calibration']


def test_derived_data_fit_train_only_preserves_raw_and_rejects_mutated_cache(tmp_path,monkeypatch):
    root=tmp_path/'repo';source=root/'data/toy';source.mkdir(parents=True)
    configs=root/'configs/datasets';configs.mkdir(parents=True)
    train=np.column_stack([np.arange(13.),np.arange(13.)*2]);train[10:]+=1000
    test=np.ones((12,2))*2000;labels=np.zeros(12)
    for name,array in [('train.npy',train),('test.npy',test),('label.npy',labels)]:np.save(source/name,array)
    ds=dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='label.npy'))
    (configs/'mtsad_smd.json').write_text(json.dumps(ds))
    before={p.name:subject.sha(p) for p in source.iterdir()}
    config=dict(data_cache_root=str(tmp_path/'cache'),data_root='data',contract_sha256='frozen')
    contract=dict(reported_table13_splits=dict(SMD=dict(train=10,validation=3,test=12,features=2)))
    monkeypatch.setattr(subject,'ROOT',root)
    folder,manifest=subject.prepare_data(config,'SMD',contract)
    assert subject.read(folder/'scaler.json')['fitted_rows']==10
    assert np.allclose(np.load(folder/'train.npy').mean(0),0,atol=1e-6)
    assert np.load(folder/'validation.npy').min()>100
    assert before=={p.name:subject.sha(p) for p in source.iterdir()}
    subject.prepare_data(config,'SMD',contract)
    with (folder/'validation.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='cache changed'):subject.prepare_data(config,'SMD',contract)


def test_full_width_training_scoring_and_controlled_validation_integration(tmp_path):
    code=r'''
import json,pathlib,sys,numpy as np
from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as s
root=pathlib.Path(sys.argv[1]);raw=root/'data/toy';raw.mkdir(parents=True)
dsroot=root/'configs/datasets';dsroot.mkdir(parents=True)
rng=np.random.default_rng(1103)
np.save(raw/'train.npy',rng.normal(size=(500,3)));np.save(raw/'test.npy',rng.normal(size=(100,3)))
labels=np.zeros(100,dtype=np.int8);labels[20:40]=1;np.save(raw/'labels.npy',labels)
(dsroot/'mtsad_smd.json').write_text(json.dumps(dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='labels.npy'))))
contract=dict(reported_table13_splits=dict(SMD=dict(train=400,validation=100,test=100,features=3)),
paper_evaluation=dict(r_percent=dict(SMD=.5),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
source=s.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
s.ROOT=root;s.verify=lambda _:contract
config=dict(artifact_root=str(root/'artifacts'),data_cache_root=str(root/'cache'),data_root='data',
source_directory=str(source),contract_sha256='fixture',explicit_reconstruction_choices=['fixture test only'],boundary='fixture only, not benchmark performance')
path=root/'config.json';path.write_text(json.dumps(config))
s.run_worker(config,dict(id='fixture_full_width',dataset='SMD',seed=1103),path)
r=json.loads((root/'artifacts/fixture_full_width/result.json').read_text())
assert r['parameter_count']>4000000 and 1<=len(r['epochs'])<=10
assert r['optimizer_updates']==len(r['epochs']) and r['actual_backward_calls']==2*r['optimizer_updates']
assert r['covered_points']==dict(train=400,validation=100,test=100)
assert len(r['metrics'])==7 and not r['checkpoint_selection_uses_test'] and not r['threshold_calibration_uses_test']
for split,n in [('train',400),('validation',100),('test',100)]:
 with np.load(root/'artifacts/fixture_full_width'/(split+'_scores.npz')) as a:
  assert a['component.layer_discrepancies'].shape==(n,3)
  assert all(a[k].shape==(n,) for k in s.STREAMS)
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(tmp_path)],cwd=subject.ROOT,
                          env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-4000:]


def test_controller_import_is_numerically_cold():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_paper_contract_v1;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
