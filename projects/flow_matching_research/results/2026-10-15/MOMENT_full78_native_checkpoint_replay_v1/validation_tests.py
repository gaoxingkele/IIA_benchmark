"""Full MOMENT bindings,padding/masks,timestamps and independent scoring checks."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as subject
from scripts.flow_matching.register_moment_full_checkpoint_replay_v1 import registration


def test_all72_original_and6_controls_are_separate_complete_cases():
    runtime=registration();cases=subject.verify(runtime)
    assert len(cases)==78 and sum(c['family']=='original' for c in cases.values())==72
    assert sum(c['family']=='constant_fit' for c in cases.values())==6
    for family in ['original','constant_fit']:
        selected=[c for c in cases.values() if c['family']==family]
        assert {c['job']['window'] for c in selected}=={100,512}
        assert len({c['config']['parameters']['checkpoint_sha256'] for c in selected})==3
    assert {c['job']['seed'] for c in cases.values()}=={42}


@pytest.mark.parametrize('change',['missing','guard','caps','tolerance','bytes','scope'])
def test_rejects_partial_or_relaxed_native_replay(change):
    c=registration()
    if change=='missing':c['required_ids'].pop()
    elif change=='guard':c['settings']['minimum_commit_headroom_bytes']=28*1024**3
    elif change=='caps':c['settings']['maximum_worker_private_bytes']=20*1024**3
    elif change=='tolerance':c['mse_tolerance']['atol']=1e-3
    elif change=='bytes':c['array_comparison']='mean_only'
    else:c['original_scope']['papers']=1
    with pytest.raises(ValueError):subject.verify(c)


def test_standard_and_industrial_group_orders_preserve_source_boundaries():
    standard=dict(entities=[dict(entity_id='e1',array_prefix='a'),dict(entity_id='e2',array_prefix='b')])
    assert [key for _,_,key in subject.groups(dict(input_kind='standard'),standard)]==['a_train','a_validation','a_test','b_train','b_validation','b_test']
    industrial=dict(training_groups=[dict(group_id='tr',array_prefix='t')],validation_groups=[dict(group_id='va',array_prefix='v')],entities=[dict(entity_id='test',array_prefix='z')])
    assert [key for _,_,key in subject.groups(dict(input_kind='industrial'),industrial)]==['t_values','v_values','z_test']


def test_historical_guard_is_not_retroactively_certified(tmp_path,monkeypatch):
    monkeypatch.setattr(subject,'ROOT',tmp_path)
    subject.write(tmp_path/'queue.json',dict(fixture=True))
    runtime=dict(source_guard_directories=dict(original=['receipts']))
    case=dict(family='original',path='queue.json',job=dict(id='fixture'))
    assert not subject.guard_history(runtime,case)['original_execution_guard_history_proven']
    subject.write(tmp_path/'receipts/fixture.json',dict(job_id='fixture',exit_code=0,resource_guard_aborted=False,
        full_registered_budget_retained=True,queue_sha256=subject.sha(tmp_path/'queue.json')))
    assert subject.guard_history(runtime,case)['original_execution_guard_history_proven']
    d=subject.read(tmp_path/'receipts/fixture.json');d['resource_guard_aborted']=True;subject.write(tmp_path/'receipts/fixture.json',d)
    with pytest.raises(ValueError):subject.guard_history(runtime,case)


@pytest.mark.parametrize('window,policy',[(100,'TAB_always_extra_patch'),(512,'minimal_patch')])
def test_actual_source_padding_full_windows_and_observed_MSE_in_producer_environment(tmp_path,window,policy):
    # Toy frozen neural module only. Production uses the registered official weights.
    code=r'''
import numpy as np,torch,sys
from types import SimpleNamespace
from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import observed_score
from iia_benchmark.models.fm_moment_pretrained import reconstruct_windows
from scripts.flow_matching.run_moment_pretrained_job import segment_scores
window=int(sys.argv[1]);policy=sys.argv[2];torch.set_num_threads(2)
class Fixture(torch.nn.Module):
 patch_len=8
 def __init__(self):super().__init__();self.calls=[]
 def forward(self,**kw):
  x=kw['x_enc'];self.calls.append(tuple(x.shape))
  assert torch.equal(kw['input_mask'],torch.ones_like(kw['input_mask'])) and torch.equal(kw['mask'],torch.ones_like(kw['mask']))
  return SimpleNamespace(reconstruction=x*.75)
net=Fixture().eval()
class Adapter:
 def __init__(self):self.model=net
 def score(self,x):return reconstruct_windows(self.model,x,1,'cpu',policy)
adapter=Adapter();checked,record=observed_score(adapter,window,dict(rtol=2e-6,atol=2e-6));adapter.score=checked
for length in [window-3,window,window+7,2*window+3]:
 data=np.arange(length*3,dtype=np.float32).reshape(length,3)/1000
 values,audit=segment_scores(adapter,data,window)
 expected=np.square(data.astype(np.float64)*.25).mean(axis=1)
 assert values.shape==(length,) and np.allclose(values,expected,rtol=2e-6,atol=2e-6)
 assert audit['native_points']==length and audit['short_segment_replicated_points']==max(window-length,0)
assert record['windows']==record['forward_calls']==len(net.calls) and record['window_points']==record['windows']*window
expected_padding=(-window)%8 if policy=='minimal_patch' else 8-window%8
assert all(shape[-1]==window+expected_padding for shape in net.calls)
print('all_source_windows_padding_and_MSE_checked;not_benchmark_performance')
'''
    python=subject.ROOT/'.venv/moment-author-py310/Scripts/python.exe'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             PYTHONPATH=str(subject.ROOT)+os.pathsep+str(subject.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subprocess.run([str(python),'-B','-X','utf8','-c',code,str(window),policy],cwd=subject.ROOT,env=env,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout[-1000:]+p.stderr[-3500:]


def test_independent_all_controls_handle_ties_multiple_entities_and_single_class_entities():
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    entities={'a':(np.array([0,1,1,0]),np.array([.1,.5,.5,.1])),
              'positive_only':(np.ones(3,np.int8),np.array([.1,.1,.8])),
              'normal_only':(np.zeros(3,np.int8),np.array([.1,.5,.5]))}
    validation=np.array([.1,.2,.3,.4]);training=np.array([.1,.2,.3])
    evaluation=dict(strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,bootstrap_resamples=200,
        bootstrap_block_points=1024,tab_threshold_ratios_percent=[.1,.5,1,2,3,5,10,15,20,25],tab_commit='fixture')
    actual=evaluate_saved(entities,validation,training,'a'*64,'b'*64,evaluation,42)
    controls=subject.independent_controls(entities,validation,training,actual,evaluation)
    assert set(controls)=={'0.1','1','5'}
    assert controls['1']['per_entity']['positive_only']['average_precision']==1
    assert controls['1']['per_entity']['normal_only']['average_precision'] is None
    changed=deepcopy(actual);changed['tab_core_diagnostic']['ratio_grid'][0]['threshold']+=1
    with pytest.raises(ValueError):subject.independent_controls(entities,validation,training,changed,evaluation)


def test_metadata_registration_and_guard_API_remain_numerically_cold():
    code=('import sys;from scripts.flow_matching.register_moment_full_checkpoint_replay_v1 import registration;'
        'from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import verify;assert len(verify(registration()))==78;'
        'from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait;'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr[-4000:]


@pytest.mark.parametrize('kind',['standard','industrial'])
def test_full_source_native_and_independent_worker_bodies_in_producer_environment(tmp_path,kind):
    # Full-body toy checkpoint fixture; never official-weight benchmark evidence.
    code=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import run_moment_pretrained_job as original
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as s
from iia_benchmark.models import fm_moment_pretrained as adapter
from types import SimpleNamespace
root=pathlib.Path(sys.argv[1]);kind=sys.argv[2];s.ROOT=root;original.ROOT=root
torch.set_num_threads(2)
class Fixture(torch.nn.Module):
 patch_len=8
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.75))
 def forward(self,**kw):return SimpleNamespace(reconstruction=kw['x_enc']*self.weight)
def frozen_load(*args):
 model=Fixture().eval();model.requires_grad_(False)
 audit=dict(checkpoint_sha256='a'*64,config_sha256='b'*64,state_tensors=1,all_keys_strictly_loaded=True,
  parameter_count=1,pretrained_size='fixture_only',configured_sequence_length=100,task='reconstruction',
  random_reinitialized_head=False,hub_downloads=False)
 return model,audit
adapter.load_pretrained=frozen_load
config=dict(entrypoint='iia_benchmark.models.fm_moment_pretrained:MomentPretrainedWindow',parameters=dict(
 source_root='fixture_source',checkpoint_directory='fixture_local_reference',checkpoint_sha256='a'*64,config_sha256='b'*64,
 window=100,batch_size=1,padding_policy='TAB_always_extra_patch',epochs=0,seed=42,device='cpu'),reproduction_status='fixture_only',boundary='fixture_not_benchmark',repeat_boundary='fixture_only')
s.write(root/'model.json',config)
job=dict(id='fixture',dataset='fixture',seed=42,window=100,model_config='model.json',dataset_manifest_path='manifest.json',
 input_kind=kind,parameter_overrides=dict(device='cpu',seed=42),output_directory='source_run',frozen_files=[])
if kind=='standard':
 dataset=dict(id='fixture',entities=[dict(entity_id='e1',array_prefix='a'),dict(entity_id='e2',array_prefix='b')])
else:
 dataset=dict(id='fixture',training_groups=[dict(group_id='train',array_prefix='a')],validation_groups=[dict(group_id='validation',array_prefix='b')],entities=[dict(entity_id='test1',array_prefix='c'),dict(entity_id='test2',array_prefix='d')])
payload={};rng=np.random.default_rng(42)
for n,(role,group,key) in enumerate(s.groups(job,dataset)):
 length=[97,100,207,203,101,99][n%6];prefix=group['array_prefix']
 payload[key]=rng.normal(size=(length,3)).astype(np.float32)
 payload[prefix+'_timestamps']=np.arange(length,dtype=np.int64)
 payload[prefix+'_source_rows']=np.arange(length,dtype=np.int64)+10
 if role=='test':
  y=np.zeros(length,np.int8);y[10:30]=1;payload[prefix+'_labels']=y
np.savez_compressed(root/'inputs.npz',**payload)
dataset.update(input_npz='inputs.npz',input_sha256=s.sha(root/'inputs.npz'),grouped_splits_verified=True)
s.write(root/'manifest.json',dict(datasets=[dataset]))
evaluation=dict(cpu_threads=2,strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,bootstrap_resamples=200,
 bootstrap_block_points=1024,tab_threshold_ratios_percent=[.1,.5,1,2,3,5,10,15,20,25],tab_commit='fixture_only')
original.run(job,evaluation)
result=s.read(root/'source_run/result.json');binding=dict(result_sha256=s.sha(root/'source_run/result.json'),source_guard_history=dict(original_execution_guard_history_proven=False))
s.check_source=lambda runtime,case:(result,dataset,binding)
case=dict(job=job,config=config,queue=dict(evaluation=evaluation),family='fixture')
s.write(root/'runtime.json',dict(fixture_only=True))
runtime=dict(artifact_root=str(root/'replay'),_config_path='runtime.json',mse_tolerance=dict(rtol=2e-6,atol=2e-6))
s.native_worker(runtime,case);s.review_worker(runtime,case)
native=s.read(root/'replay/fixture/native/native_replay.json');review=s.read(root/'replay/fixture/independent/review.json')
assert native['passed'] and review['passed'] and len(native['native_coverage'])==len(s.groups(job,dataset))
assert native['inline_full_window_MSE_check']['windows']==sum(r['windows'] for r in native['native_coverage'])
assert not review['new_training_seed'] and not review['historical_paper_equivalence_certified']
assert not review['original_source_guard_history']['original_execution_guard_history_proven']
'''
    python=subject.ROOT/'.venv/moment-author-py310/Scripts/python.exe'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(subject.ROOT)+os.pathsep+str(subject.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    p=subprocess.run([str(python),'-B','-X','utf8','-c',code,str(tmp_path),kind],cwd=subject.ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout[-1500:]+p.stderr[-4500:]
