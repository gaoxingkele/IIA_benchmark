"""Full source-loop adapters;all numerical examples below are fixtures only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import run_moment_icm_full_forecast_v1 as s


def test_all54_cases_and_five_epoch_source_recipe_are_frozen():
    c=s.registration();jobs=s.verify(c)
    assert len(jobs)==54 and len({j['dataset'] for j in jobs.values()})==6
    assert {j['branch'] for j in jobs.values()}==set(s.f.BRANCHES)
    assert {j['horizon'] for j in jobs.values()}=={96,192,384}
    assert all(j['epochs']==5 and j['seed']==13 for j in jobs.values())
    assert c['source_recipe']['num_workers']==5 and c['source_recipe']['shuffle']
    assert c['source_recipe']['train_batch_size']==64 and c['source_recipe']['val_batch_size']==256
    assert c['source_recipe']['head_dropout']==.1 and c['source_recipe']['add_positional_embedding'] is False
    assert c['settings']==s.read(s.ROOT/s.f.pretrained.pretrain.CONFIG)['settings']
    assert c['original_scope']['paper_scalar_targets']==702


@pytest.mark.parametrize('change',['jobs','source_recipe','settings','prediction_tolerance','both_full_pretrained_models_required'])
def test_reduced_scope_epochs_precision_resources_or_checks_cannot_enter(change):
    c=s.registration()
    if change=='jobs':c['jobs'].pop()
    elif change=='source_recipe':c['source_recipe']['max_epoch']=1
    elif change=='settings':c['settings']['minimum_gpu_free_bytes']=1
    elif change=='prediction_tolerance':c[change]['atol']=.1
    else:c[change]=False
    with pytest.raises(ValueError,match='All54'):s.verify(c)


def native(code,args=()):
    r=subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,*map(str,args)],
        cwd=s.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
        capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout[-1500:]+r.stderr[-5000:]


@pytest.mark.parametrize('branch',['independent_head','icm_beta_frozen','icm_beta_finetuned'])
def test_real_full_tiny_head_and_beta_gradients_with_frozen_backbone(branch):
    native(r'''
import sys,torch
from scripts.flow_matching import moment_icm_forecast_finetune_v1 as f
torch.set_num_threads(2);torch.manual_seed(13);branch=sys.argv[1]
cfg=f.read(f.ROOT/f.pretrained.pretrain.masks.TRAIN_BODY)
p=dict(f.read(f.ROOT/cfg['core_model_configuration'])['parameters'],mixer=f.BRANCHES[branch]['mixer'])
core=f.pretrained.pretrain.train.core_class(f.ROOT/'src/iia_benchmark/models')(**p)
assert core.patch_embedding.add_positional_embedding is False
model=f.make_forecaster(core,96,branch,f.ROOT/f.SOURCE)
frozen={n:t.detach().clone() for n,t in model.named_parameters() if not t.requires_grad}
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0)
x=torch.randn(2,3,256);mask=torch.ones(2,256,dtype=torch.long);target=torch.randn(2,3,96)
out=model(x_enc=x,input_mask=mask).forecast;assert out.shape==target.shape
loss=(out-target).square().mean();loss.backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters() if t.requires_grad)
assert all(t.grad is None for t in model.parameters() if not t.requires_grad)
optimizer.step()
for name,t in model.named_parameters():
 if name in frozen:assert torch.equal(t,frozen[name])
assert sum(t.numel() for t in model.parameters() if t.requires_grad)==96*8193+(16 if branch=='icm_beta_finetuned' else 0)
''',[branch])


def test_exact_source_training_and_validation_AST_changes_are_only_declared():
    native(r'''
import ast,copy
from scripts.flow_matching import moment_icm_forecast_finetune_v1 as f
ns,methods=f.source_symbols(f.ROOT/f.SOURCE)
tree=ast.parse((f.ROOT/f.SOURCE/'moment/tasks/forecast_finetune.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ForecastFinetuning')
for name in ['train','validation','evaluate_and_log','compute_forecasting_metrics']:
 actual=copy.deepcopy(next(n for n in methods if n.name==name))
 original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 if name=='train':
  found=0
  for loop in [n for n in ast.walk(actual) if isinstance(n,ast.For)]:
   found+=sum(ast.unparse(x)=='self.audit_gradients(loss)' for x in loop.body)
   loop.body=[x for x in loop.body if ast.unparse(x)!='self.audit_gradients(loss)']
  assert found==1
 if name=='validation':actual.body[0]=copy.deepcopy(original.body[0])
 assert ast.dump(actual,include_attributes=False)==ast.dump(original,include_attributes=False)
''')


def test_actual_source_five_epoch_loop_storage_order_and_full_state_fixture(tmp_path):
    native(r'''
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,torch,yaml
from scripts.flow_matching import moment_icm_forecast_finetune_v1 as f
torch.set_num_threads(2);torch.manual_seed(13)
root=Path(sys.argv[1]);binding=dict(horizon=96,roles={})
for role in ['train','val','test']:
 d=root/role;d.mkdir();np.save(d/'values_TC.npy',np.random.RandomState(13).normal(size=(354,2)))
 np.save(d/'input_mask_window.npy',np.ones(256))
 binding['roles'][role]=dict(directory=str(d),window=dict(windows=3),facts=dict(channels=2))
loaders={role:f.RecordedLoader(torch.utils.data.DataLoader(f.WindowDataset(binding,role),batch_size=2,shuffle=True,
 num_workers=0,drop_last=False,collate_fn=f.collate_forecast),root/'run'/'orders') for role in ['train','val','test']}
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.linear=torch.nn.Linear(256,96);self.frozen=torch.nn.Parameter(torch.ones(1),requires_grad=False)
 def forward(self,x_enc,input_mask,mask=None):return SimpleNamespace(forecast=self.linear(x_enc))
args=yaml.safe_load((f.ROOT/f.SOURCE/'configs/default.yaml').read_text())
args.update(yaml.safe_load((f.ROOT/f.SOURCE/'configs/forecasting/linear_probing.yaml').read_text()))
args.update(use_amp=False,forecast_horizon=96,seq_len=256,dataset_names='fixture.csv',checkpoint_path=str(root/'run'/'checkpoints'))
task=f.SourceForecastTask(Toy(),loaders,SimpleNamespace(**args),root/'run',f.ROOT/f.SOURCE,
 dict(id='fixture_only_not_benchmark',scientific=False),'cpu',dict(fixture=True))
task.train();proof=task.finalize()
assert proof['full_epochs']==5 and proof['full_optimizer_updates']==10
assert len([r for r in loaders['train'].records if r['phase']=='train'])==5
assert len([r for r in loaders['train'].records if r['phase']=='validation'])==6
assert len(loaders['val'].records)==6
bundle=proof['selected_TEST_bundle'];p=np.load(Path(bundle['directory'])/'prediction.npy');y=np.load(Path(bundle['directory'])/'truth.npy')
order=np.load(bundle['order']['order_path']);assert p.shape==y.shape==(3,2,96)
stats=f.independent_metrics(p,y,order,block=352)
assert stats['coordinates']==576 and stats['conditional_block_bootstrap']['training_seed_uncertainty'] is False
assert proof['frozen_parameters_exactly_unchanged']
''',[tmp_path])


def test_spawn_safe_readonly_loader_has_every_origin_and_partial_batch(tmp_path):
    native(r'''
from pathlib import Path
import sys,numpy as np
from scripts.flow_matching import moment_icm_forecast_finetune_v1 as f
root=Path(sys.argv[1]);binding=dict(horizon=96,roles={})
for role in ['train','val','test']:
 d=root/role;d.mkdir();np.save(d/'values_TC.npy',np.arange(364*2,dtype=float).reshape(364,2))
 np.save(d/'input_mask_window.npy',np.ones(256));binding['roles'][role]=dict(directory=str(d),window=dict(windows=13),facts=dict(channels=2))
loaders=f.make_loaders(binding,root/'orders',num_workers=1)
for loader in loaders.values():
 batches=list(loader);assert len(batches)==1 and len(batches[0].window_indices)==13
 assert loader.records[-1]['last_batch_items']==13 and loader.records[-1]['every_origin_once']
''',[tmp_path])


def test_scientific_worker_requires_actual_full_models_before_numerical_imports(monkeypatch,tmp_path):
    monkeypatch.setattr(s.f.pretrained,'trained_checkpoint',lambda *a:None)
    monkeypatch.setattr(s.f,'ready_data',lambda *a:None)
    with pytest.raises(ValueError,match='before NN imports'):
        s.f.worker({},dict(branch='independent_head',dataset='Exchange',horizon=96),tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_cold_registration_has_no_numpy_torch_or_actual_performance_claim():
    code='''
import sys
from scripts.flow_matching import run_moment_icm_full_forecast_v1 as s
c=s.registration();s.verify(c)
assert len(c['jobs'])==54 and 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout[-1000:]+r.stderr[-3000:]
