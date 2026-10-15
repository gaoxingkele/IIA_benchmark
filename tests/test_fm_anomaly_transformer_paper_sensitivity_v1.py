"""Reported-axis completeness and exact frozen-worker parameter binding."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_paper_sensitivity_v1 as subject


def config_fixture(monkeypatch):
    parent=subject.read(subject.ROOT/subject.PARENT)
    contract=subject.read(subject.ROOT/subject.parent.CONTRACT)
    monkeypatch.setattr(subject.parent,'verify',lambda _:contract)
    config=dict(parent,parent_registration=subject.PARENT,parent_registration_sha256=subject.sha(subject.ROOT/subject.PARENT),
                axes=subject.AXES,base=subject.BASE,base_model_reuse=parent['jobs'],
                primary_score_stream='all.multiplication',score_stream_policy='all_four_criteria_plus_every_individual_layer',
                score_streams_by_layers={str(n):subject.streams(n) for n in subject.AXES['layers']},
                source_receipts=[])
    config['jobs']=[]
    for axis,value in subject.variants():
        for job in parent['jobs']:
            values=dict(subject.BASE);values[axis]=value
            config['jobs'].append(dict(id=f"anomaly_transformer__paper_sensitivity__{axis}{value}__{job['dataset'].lower()}__seed{job['seed']}",
                                      dataset=job['dataset'],seed=job['seed'],axis=axis,value=value,parameters=values))
    return config


def test_all_reported_nonbase_axes_are_registered_without_retraining_base(monkeypatch):
    c=config_fixture(monkeypatch);subject.verify(c)
    assert len(subject.variants())==16 and len(c['jobs'])==400 and len(c['base_model_reuse'])==25
    assert len({j['id'] for j in c['jobs']})==400
    assert all(sum(j['parameters'][k]!=subject.BASE[k] for k in subject.BASE)==1 for j in c['jobs'])


@pytest.mark.parametrize('mutation',['window_endpoint','weight_zero','depth_four','hidden1024','job','base','guards','epochs','primary','scoring'])
def test_full_scope_budgets_and_primary_criterion_cannot_be_redefined(monkeypatch,mutation):
    c=deepcopy(config_fixture(monkeypatch))
    if mutation=='window_endpoint':c['axes']['window'].pop()
    elif mutation=='weight_zero':c['axes']['weight'].remove(0)
    elif mutation=='depth_four':c['axes']['layers'].remove(4)
    elif mutation=='hidden1024':c['axes']['hidden'].remove(1024)
    elif mutation=='job':c['jobs'].pop()
    elif mutation=='base':c['base_model_reuse'].pop()
    elif mutation=='guards':c['settings']['minimum_commit_headroom_bytes']=32*1024**3
    elif mutation=='epochs':c['training']['maximum_epochs']=3
    elif mutation=='primary':c['primary_score_stream']='choose_best_test_F1'
    else:c['score_stream_policy']='only_primary'
    with pytest.raises(ValueError):subject.verify(c)


@pytest.mark.parametrize('axis,value',subject.variants())
def test_each_paper_axis_changes_only_its_selected_scientific_value(axis,value):
    values=dict(subject.BASE);values[axis]=value
    worker=subject.bound_worker(values,verify_function=lambda _:None)
    namespace=worker.__globals__
    assert namespace['SELECTED_WINDOW']==values['window']
    assert namespace['SELECTED_HIDDEN']==values['hidden']
    assert namespace['SELECTED_LAYERS']==values['layers']
    assert namespace['minimax_losses'].keywords==dict(weight=values['weight'])
    assert len(namespace['STREAMS'])==4+values['layers']
    assert namespace['score_components'] is subject.parent.score_components
    assert namespace['prepare_data'] is subject.parent.prepare_data


def test_zero_weight_reduces_both_training_phases_to_reconstruction():
    import torch
    p=torch.tensor([[[[.8,.2],[.5,.5]]]],requires_grad=True)
    s=torch.full_like(p,.5,requires_grad=True)
    x=torch.ones(1,2,2);y=torch.zeros_like(x,requires_grad=True)
    minimum,maximum,rec=subject.parent.minimax_losses(y,x,[p],[s],weight=0)
    assert torch.equal(minimum,rec) and torch.equal(maximum,rec)
    gradient=torch.autograd.grad(minimum,[p],allow_unused=True)[0]
    assert gradient is not None and torch.equal(gradient,torch.zeros_like(gradient))


def test_actual_bound_training_has_selected_shapes_full_budget_and_score_coverage(tmp_path):
    code=r'''
import json,pathlib,sys,numpy as np
from scripts.flow_matching import run_anomaly_transformer_paper_sensitivity_v1 as s
root=pathlib.Path(sys.argv[1]);raw=root/'data/toy';raw.mkdir(parents=True)
configs=root/'configs/datasets';configs.mkdir(parents=True)
rng=np.random.default_rng(1103)
np.save(raw/'train.npy',rng.normal(size=(300,3)));np.save(raw/'test.npy',rng.normal(size=(150,3)))
labels=np.zeros(150,dtype=np.int8);labels[20:40]=1;np.save(raw/'labels.npy',labels)
(configs/'mtsad_smd.json').write_text(json.dumps(dict(loader='npy',directory='toy',files=dict(train='train.npy',test='test.npy',label='labels.npy'))))
contract=dict(reported_table13_splits=dict(SMD=dict(train=200,validation=100,test=150,features=3)),paper_evaluation=dict(r_percent=dict(SMD=.5),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
source=s.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
s.parent.ROOT=root
config=dict(artifact_root=str(root/'artifacts'),data_cache_root=str(root/'cache'),data_root='data',source_directory=str(source),contract_sha256='fixture',explicit_reconstruction_choices=['fixture only'],boundary='fixture only, not benchmark performance')
path=root/'config.json';path.write_text(json.dumps(config));job=dict(id='fixture',dataset='SMD',seed=1103)
# Compound small fixture proves all four bindings execute; production verify
# accepts only the400 registered single-factor jobs.
worker=s.bound_worker(dict(window=50,weight=0,layers=1,hidden=256),verify_function=lambda _:contract,root=root)
worker(config,job,path)
r=json.loads((root/'artifacts/fixture/result.json').read_text())
assert r['parameter_count']<2000000 and 4<=len(r['epochs'])<=10
assert r['optimizer_updates']==len(r['epochs']) and r['actual_backward_calls']==2*r['optimizer_updates']
assert r['covered_points']==dict(train=200,validation=100,test=150)
assert len(r['metrics'])==5 and not r['checkpoint_selection_uses_test']
assert all(e['train_losses'][0]==e['train_losses'][1] for e in r['epochs'])
with np.load(root/'artifacts/fixture/test_scores.npz') as a:
 assert a['component.layer_discrepancies'].shape==(150,1)
 assert 'layer1.multiplication' in a.files and 'layer2.multiplication' not in a.files
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(tmp_path)],cwd=subject.ROOT,env=env,
                          capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-4000:]


def test_numerically_cold_controller_import():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_paper_sensitivity_v1;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
