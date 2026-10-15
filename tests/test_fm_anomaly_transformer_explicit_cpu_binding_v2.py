"""Verify explicit Windows CPU binding without changing frozen numerical code."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_explicit_cpu_binding_v2 as subject


def test_environment_override_is_narrow_and_does_not_mutate_caller(monkeypatch):
    records=[]
    monkeypatch.setattr(subject.subprocess,'Popen',lambda command,**kw:records.append((command,kw)) or 'child')
    environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',UNCHANGED='yes')
    assert subject.bound_popen(['python','worker'],env=environment,cwd='work')=='child'
    assert environment['CUDA_VISIBLE_DEVICES']==''
    assert records==[(['python','worker'],dict(env=dict(environment,CUDA_VISIBLE_DEVICES='-1'),cwd='work'))]


@pytest.mark.parametrize('branch',['author','restored'])
def test_real_source_solver_is_explicitly_CPU_even_on_GPU_host(branch,tmp_path):
    original=subject.ROOT/'experiments/runs/fm_baseline_code_completion/sources/anomaly_transformer_author/original'
    workspace=tmp_path/'source';shutil.copytree(original,workspace,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    patch=subject.author.patched if branch=='author' else subject.restored.patched
    for name in ['model/attn.py','solver.py']:
        p=workspace/name;p.write_text(patch(name,p.read_text(encoding='utf-8')),encoding='utf-8')
    code=r'''
import sys,pathlib,numpy as np,pandas as pd,torch
assert not torch.cuda.is_available(), 'CPU mask must disable CUDA in the actual isolated child'
torch.set_num_threads(2);torch.manual_seed(1103)
from solver import Solver
data=pathlib.Path('fixture');data.mkdir();rng=np.random.default_rng(1103)
branch=sys.argv[1];features=38 if branch=='author' else 51;dataset='SMD' if branch=='author' else 'SWaT'
train=rng.normal(size=(100,features));test=rng.normal(size=(100,features));labels=np.zeros(100);labels[40:60]=1
if branch=='author':
 np.save(data/'SMD_train.npy',train);np.save(data/'SMD_test.npy',test);np.save(data/'SMD_test_label.npy',labels)
else:
 pd.DataFrame(np.column_stack([train,np.zeros(100)])).to_csv(data/'swat_train2.csv',index=False)
 pd.DataFrame(np.column_stack([test,labels])).to_csv(data/'swat2.csv',index=False)
s=Solver(dict(lr=.0001,num_epochs=1,k=3,win_size=100,input_c=features,output_c=features,
batch_size=256,dataset=dataset,data_path=str(data),model_save_path='checkpoints',anormly_ratio=.1))
assert s.device.type=='cpu' and all(p.device.type=='cpu' for p in s.model.parameters())
s.train();s.test()
assert all(p.device.type=='cpu' for p in s.model.parameters())
assert sum(p.numel() for p in s.model.parameters())>4000000
'''
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subject.bound_popen([sys.executable,'-B','-X','utf8','-c',code,branch],cwd=workspace,
                          env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=p.communicate(timeout=100)
    assert p.returncode==0,stdout[-2000:]+stderr[-3000:]


def test_binding_import_is_numerically_cold():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_explicit_cpu_binding_v2;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr
