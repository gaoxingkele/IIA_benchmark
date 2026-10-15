"""Recovery fidelity tests; fixture computations are not benchmark results."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_swat_restored_v1 as subject


def config_fixture(tmp_path):
    parent=subject.read(subject.ROOT/'configs/experiments/fm_anomaly_transformer_author_full.v1.json')
    parent['source_receipts']=[]
    path=tmp_path/'parent.json';path.write_text(json.dumps(parent),encoding='utf-8')
    config=dict(parent,parent_original_matrix='parent.json',parent_original_matrix_sha256=subject.sha(path),
                loader_source=subject.LOADER_SOURCE,training_rows_policy='all_495000_payload_rows_no_trim',
                loader_restoration_author_equivalence_certified=False)
    config['jobs']=[dict(id=f'anomaly_transformer_author__restored_swat_loader__swat__seed{s}',
        dataset='SWaT',seed=s,epochs=3,anomaly_ratio=.1,dataset_config='configs/datasets/mtsad_swat.json',
        status='registered_full_author_runtime') for s in parent['seeds']]
    return config


def test_complete_restoration_matrix_keeps_parent_scope(tmp_path):
    assert subject.verify(config_fixture(tmp_path),tmp_path)


@pytest.mark.parametrize('mutation',['seed','missing_job','epoch','ratio','width','guard','trim','equivalence','scope','parent'])
def test_rejects_budget_protocol_identity_and_false_equivalence(mutation,tmp_path):
    config=config_fixture(tmp_path)
    if mutation=='seed':config['jobs'][0]['seed']=42
    elif mutation=='missing_job':config['jobs'].pop()
    elif mutation=='epoch':config['jobs'][0]['epochs']=1
    elif mutation=='ratio':config['jobs'][0]['anomaly_ratio']=1
    elif mutation=='width':config['required_model']=dict(config['required_model'],d_model=32)
    elif mutation=='guard':config['settings']=dict(config['settings'],minimum_commit_headroom_bytes=32*1024**3)
    elif mutation=='trim':config['training_rows_policy']='guess_475200_rows'
    elif mutation=='equivalence':config['loader_restoration_author_equivalence_certified']=True
    elif mutation=='scope':config['original_scope']={}
    else:config['parent_original_matrix_sha256']='0'*64
    with pytest.raises(ValueError):subject.verify(config,tmp_path)


def test_extracted_loader_class_is_exact_pinned_source():
    whole=(subject.ROOT/subject.LOADER_SOURCE).read_text(encoding='utf-8')
    original=next(n for n in ast.parse(whole).body if isinstance(n,ast.ClassDef) and n.name=='SWATSegLoader')
    extracted=ast.parse(subject.loader_code()).body[0]
    assert ast.dump(original,include_attributes=False)==ast.dump(extracted,include_attributes=False)


def test_solver_retains_all_original_numerical_updates_and_explicit_restoration():
    source=(subject.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original/solver.py').read_text(encoding='utf-8')
    output=subject.patched('solver.py',source);ast.parse(output)
    for line in ['loss1.backward(retain_graph=True)','loss2.backward()',
                 'self.optimizer.step()','self.vali(self.test_loader)',
                 'EarlyStopping(patience=3','np.percentile(combined_energy, 100 - self.anormly_ratio)']:
        assert output.count(line)==source.count(line)
    assert "step=1, flag=mode" in output and 'num_workers=0' in output
    assert "dataset != 'SWaT'" in output


def test_actual_loader_scaling_coverage_and_full_width_source_training(tmp_path):
    original=subject.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
    workspace=tmp_path/'source';shutil.copytree(original,workspace,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['model/attn.py','solver.py']:
        p=workspace/name;p.write_text(subject.patched(name,p.read_text(encoding='utf-8')),encoding='utf-8')
    code=r'''
import hashlib,pathlib,numpy as np,pandas as pd,torch
torch.set_num_threads(2);torch.manual_seed(1103)
from solver import Solver, get_loader_segment
data=pathlib.Path('fixture');data.mkdir();rng=np.random.default_rng(1103)
train=rng.normal(size=(100,51));test=rng.normal(size=(149,51))
train_labels=np.zeros(100);test_labels=np.zeros(149);test_labels[50:80]=1
pd.DataFrame(np.column_stack([train,train_labels])).to_csv(data/'swat_train2.csv',index=False)
pd.DataFrame(np.column_stack([test,test_labels])).to_csv(data/'swat2.csv',index=False)
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
loader=get_loader_segment(str(data),256,100,mode='thre',dataset='SWaT')
assert len(loader.dataset)==1 and len(loader.dataset.test)==149
assert loader.dataset.train.shape==(100,51) and loader.dataset.test.shape==(149,51)
assert np.allclose(loader.dataset.train.mean(axis=0),0,atol=1e-12)
assert np.array_equal(loader.dataset[0][1].reshape(-1),test_labels[:100])
try:get_loader_segment(str(data),256,100,mode='train',dataset='PSM')
except ValueError:pass
else:raise AssertionError('Silently replaced another dataset loader')
# One full-width source update verifies integration only; no benchmark claim.
s=Solver(dict(lr=.0001,num_epochs=1,k=3,win_size=100,input_c=51,output_c=51,
batch_size=256,dataset='SWaT',data_path=str(data),model_save_path='checkpoints',anormly_ratio=.1))
assert s.model.embedding.value_embedding.tokenConv.out_channels==512
assert len(s.model.encoder.attn_layers)==3
assert len(s.train_loader.dataset)==1 and len(s.test_loader.dataset)==50
steps=[];hook=s.optimizer.register_step_post_hook(lambda *_:steps.append(1))
s.train();hook.remove();s.test();assert len(steps)==1
with np.load('checkpoints/author_scores.npz') as a:
 assert a['test_energy'].shape==(100,) and a['train_energy'].shape==(100,)
 assert np.array_equal(a['threshold_energy'],a['test_energy'])
 assert np.isfinite(a['test_energy']).all()
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
assert before==after
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=workspace,env=env,
                          capture_output=True,text=True,timeout=100)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-3000:]


def test_controller_import_is_numerically_cold():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_swat_restored_v1;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
