"""Source/runtime fidelity and fail-closed full-budget regression checks."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_author_full_v1 as subject
from scripts.flow_matching.register_anomaly_transformer_author_full_v1 import SOURCE


def fixture_config():
    seeds = list(range(1103,1108))
    return dict(required_model=dict(d_model=512,d_ff=512,n_heads=8,e_layers=3,dropout=0.0,
        window=100,batch_size=256,k=3,learning_rate=0.0001,temperature=50),seeds=seeds,
        settings=dict(minimum_available_memory_bytes=32*1024**3,minimum_commit_headroom_bytes=40*1024**3,
                      emergency_commit_headroom_bytes=16*1024**3),source_receipts=[],
        jobs=[dict(id=f'anomaly_transformer_author__full_width__{d.lower()}__seed{s}',dataset=d,seed=s,epochs=10 if d=='SMD' else 3,
              anomaly_ratio=0.5 if d=='SMD' else (0.1 if d=='SWaT' else 1.0),
              dataset_config=f'configs/datasets/mtsad_{d.lower()}.json',
              status='source_loader_unavailable' if d=='SWaT' else 'registered_full_author_runtime')
              for d in ['SMD','MSL','SMAP','PSM','SWaT'] for s in seeds])


def test_full_matrix_is_accepted():
    assert subject.verify(fixture_config())


@pytest.mark.parametrize('field,value',[('d_model',32),('d_ff',32),('batch_size',128),('window',50),('temperature',5)])
def test_rejects_narrower_or_different_scientific_budget(field,value):
    config=fixture_config();config['required_model'][field]=value
    with pytest.raises(ValueError,match='Full original'):
        subject.verify(config)


@pytest.mark.parametrize('mutation',['missing_job','missing_seed','short_epochs','ratio','dataset','invented_swat_loader','lower_guard'])
def test_rejects_scope_protocol_and_resource_redefinitions(mutation):
    config=fixture_config()
    if mutation=='missing_job':config['jobs'].pop()
    elif mutation=='missing_seed':config['seeds'].pop()
    elif mutation=='short_epochs':config['jobs'][0]['epochs']=1
    elif mutation=='ratio':config['jobs'][0]['anomaly_ratio']=1
    elif mutation=='dataset':config['jobs'][0]['dataset_config']='other.json'
    elif mutation=='invented_swat_loader':config['jobs'][-1]['status']='registered_full_author_runtime'
    else:config['settings']['minimum_commit_headroom_bytes']=32*1024**3
    with pytest.raises(ValueError):subject.verify(config)


def test_changed_frozen_source_is_rejected(tmp_path):
    p=tmp_path/'source.py';p.write_text('original')
    config=fixture_config();config['source_receipts']=[dict(path='source.py',sha256=subject.sha(p))]
    subject.verify(config,tmp_path);p.write_text('changed')
    with pytest.raises(ValueError,match='Frozen source changed'):subject.verify(config,tmp_path)


def test_patch_is_specific_and_preserves_original_numerical_statements():
    original=(subject.ROOT/SOURCE/'solver.py').read_text(encoding='utf-8')
    output=subject.patched('solver.py',original)
    ast.parse(output)
    for statement in ['loss1.backward(retain_graph=True)','loss2.backward()',
                      'self.optimizer.step()','self.vali(self.test_loader)',
                      'EarlyStopping(patience=3','np.percentile(combined_energy, 100 - self.anormly_ratio)']:
        assert statement in output and output.count(statement)==original.count(statement)
    assert 'predictions_no_pa = pred.copy()' in output
    with pytest.raises(ValueError):subject.patched('solver.py',output)


def test_unmodified_model_and_data_loaders_are_not_reimplemented():
    for name in ['model/AnomalyTransformer.py','model/embed.py','data_factory/data_loader.py']:
        source=(subject.ROOT/SOURCE/name).read_text(encoding='utf-8')
        assert subject.patched(name,source)==source
    loader=(subject.ROOT/SOURCE/'data_factory/data_loader.py').read_text(encoding='utf-8')
    assert "elif (dataset == 'SWaT')" not in loader


def test_full_width_source_cpu_training_and_saved_arrays(tmp_path):
    """A small fixture checks actual source losses/updates; it is not performance."""
    workspace=tmp_path/'source'
    shutil.copytree(subject.ROOT/SOURCE,workspace,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['model/attn.py','solver.py']:
        p=workspace/name;p.write_text(subject.patched(name,p.read_text(encoding='utf-8')),encoding='utf-8')
    code=r'''
import json, pathlib, numpy as np, torch
torch.set_num_threads(2);torch.manual_seed(1103)
from solver import Solver
data=pathlib.Path('fixture');data.mkdir()
rng=np.random.default_rng(1103)
np.save(data/'SMD_train.npy',rng.normal(size=(200,38)))
np.save(data/'SMD_test.npy',rng.normal(size=(200,38)))
labels=np.zeros(200);labels[50:80]=1;np.save(data/'SMD_test_label.npy',labels)
s=Solver(dict(lr=.0001,num_epochs=1,k=3,win_size=100,input_c=38,output_c=38,
batch_size=256,dataset='SMD',data_path=str(data),model_save_path='checkpoints',anormly_ratio=.5))
assert s.model.embedding.value_embedding.tokenConv.out_channels==512
assert len(s.model.encoder.attn_layers)==3
count=sum(p.numel() for p in s.model.parameters());assert count>4000000
steps=[];hook=s.optimizer.register_step_post_hook(lambda *_:steps.append(1))
s.train();hook.remove();assert len(steps)==1 and s.observed_epochs[0]['steps']==1
metrics=s.test()
with np.load('checkpoints/author_scores.npz') as a:
 assert a['test_energy'].shape==(200,) and a['train_energy'].shape==(200,)
 assert np.array_equal(a['threshold_energy'],a['test_energy'])
 assert np.isfinite(a['test_energy']).all() and len(metrics)==4
pathlib.Path('fixture_result.json').write_text(json.dumps(dict(parameters=count,optimizer_steps=len(steps))))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=workspace,env=env,
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-3000:]
    assert json.loads((workspace/'fixture_result.json').read_text())['optimizer_steps']==1


def test_probe_module_itself_does_not_import_numerical_libraries():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_author_full_v1;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
