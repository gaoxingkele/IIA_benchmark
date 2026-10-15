"""Full objective case scope,RNG/MC replay,path mathematics and real fixtures."""
from copy import deepcopy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import run_objective_full440_checkpoint_replay_v1 as subject
from scripts.flow_matching import objective_replay_math_v1 as independent
from scripts.flow_matching.register_objective_full440_checkpoint_replay_v1 import registration


def test_all440_cases_preserve_all_original_failures_and_deduplicate_closed_SB():
    c=registration();cases=subject.verify(c)
    assert len(cases)==440 and len(c['preserved_original_failures'])==105
    assert {k:sum(x['family']==k for x in cases.values()) for k in subject.QUEUES}==dict(original=280,repair=105,constant_fit=40,constant_fit_repair=15)
    assert c['existing_native_certificate']['id'] in cases
    assert all(cases[k]['family']=='original' for k in c['preserved_original_failures'])


@pytest.mark.parametrize('change',['missing','failure','tolerance','guard','caps','bytes'])
def test_partial_or_relaxed_contract_is_rejected(change):
    c=registration()
    if change=='missing':c['required_ids'].pop()
    elif change=='failure':c['preserved_original_failures'].pop(next(iter(c['preserved_original_failures'])))
    elif change=='tolerance':c['math_tolerance']['atol']=1e-3
    elif change=='guard':c['settings']['minimum_commit_headroom_bytes']=28*1024**3
    elif change=='caps':c['settings']['maximum_worker_private_bytes']=20*1024**3
    else:c['array_comparison']='allclose_only'
    with pytest.raises(ValueError):subject.verify(c)


@pytest.mark.parametrize('objective,sigma',[('independent',.1),('ot',.1),('target',.01),('sb',1.),('sf2m',1.),('vp',.1),('rectified',0.)])
def test_independent_every_coordinate_probability_path_equations(objective,sigma):
    import torch
    from iia_benchmark.models.fm_objective_detectors import probability_path
    rng=np.random.default_rng(1103)
    args=[torch.from_numpy(rng.normal(size=(5,200)).astype(np.float32)) for _ in range(3)]
    t=torch.tensor([.1,.3,.5,.7,.9]);a,b,e=args
    native=probability_path(a,b,t,e,objective,sigma)
    check=independent.path_numpy(a.numpy(),b.numpy(),t.numpy(),e.numpy(),objective,sigma)
    for x,y in zip(native,check):np.testing.assert_allclose(x.numpy(),y,**independent.TOLERANCE)
    same_precision=independent.path_numpy(a.numpy(),b.numpy(),t.numpy(),e.numpy(),objective,sigma,precision='float32')
    for x,y in zip(native,same_precision):np.testing.assert_allclose(x.numpy(),y,**independent.TOLERANCE)


def test_real_closed_SB_certificate_is_reused_without_new_model_inference():
    runtime=registration();cases=subject.verify(runtime);case=cases[runtime['existing_native_certificate']['id']]
    result=subject.read(subject.ROOT/case['job']['output_directory']/'result.json')
    proof=subject.reused_native(runtime,case,result)
    assert proof['full_native_reinference_reused']
    broken=deepcopy(result);broken['scores_sha256']='0'*64
    with pytest.raises(ValueError):subject.reused_native(runtime,case,broken)


def test_successful_and_failed_source_receipts_are_not_mistaken_for_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(subject,'ROOT',tmp_path)
    subject.write(tmp_path/'queue.json',{})
    case=dict(job=dict(id='fixture',output_directory='source'),family='repair',path='queue.json')
    runtime=dict(source_guard_directories=dict(repair=['receipts']))
    assert subject.source_ready(runtime,case) is False
    subject.write(tmp_path/'receipts/fixture.json',dict(job_id='fixture',queue_sha256=subject.sha(tmp_path/'queue.json'),exit_code=1))
    with pytest.raises(ValueError,match='Terminal source'):subject.source_ready(runtime,case)
    subject.write(tmp_path/'source/result.json',dict(status='completed'))
    assert subject.source_ready(runtime,case) is True # Full content checks remain inside the guarded child.


def test_full_metadata_registration_and_resource_snapshot_stay_numerically_cold():
    code=('import sys;from scripts.flow_matching.register_objective_full440_checkpoint_replay_v1 import registration;'
        'from scripts.flow_matching.run_objective_full440_checkpoint_replay_v1 import verify;assert len(verify(registration()))==440;'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=subject.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr[-4000:]


@pytest.mark.parametrize('objective,weight,stable',[('independent',1,False),('target',1,False),('sf2m',1,True),('sf2m',0,True)])
def test_actual_full20epoch_checkpoint_and_all_MC_native_independent_bodies(tmp_path,objective,weight,stable):
    # Original20-epoch/full-width scorer on tiny data. Never benchmark performance.
    code=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import run_objective_full440_checkpoint_replay_v1 as s
from scripts.flow_matching import run_tsad_experiment as source
root=pathlib.Path(sys.argv[1]);objective=sys.argv[2];weight=int(sys.argv[3]);stable=sys.argv[4]=='True';s.ROOT=root
rng=np.random.default_rng(1103);arrays={}
entities=[dict(entity_id='a',array_prefix='a'),dict(entity_id='b',array_prefix='b')]
for e in entities:
 p=e['array_prefix']
 for role,n in [('train',201),('validation',105),('test',205)]:arrays[p+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
 labels=np.zeros(205,np.int8);labels[20:50]=1;arrays[p+'_labels']=labels
 arrays[p+'_timestamps']=np.arange(205,dtype=np.int64);arrays[p+'_source_rows']=np.arange(205,dtype=np.int64)+10
np.savez_compressed(root/'inputs.npz',**arrays)
dataset=dict(id='fixture',entities=entities,input_npz='inputs.npz',input_sha256=s.sha(root/'inputs.npz'),grouped_splits_verified=True)
s.write(root/'manifest.json',dict(datasets=[dataset]))
entry='iia_benchmark.models.fm_objective_stable:StableCFMObjectiveDetector' if stable else 'iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector'
config=dict(entrypoint=entry,parameters=dict(objective=objective,sigma=1. if objective=='sf2m' else .01,
 window=100,hidden_size=128,epochs=20,batch_size=64,learning_rate=.001,score_times=[.1,.3,.5,.7,.9],monte_carlo_samples=4,score_weight=weight,seed=1103),reproduction_status='fixture_only')
s.write(root/'model.json',config)
job=dict(id='fixture',dataset='fixture',seed=1103,window=100,model_config='model.json',dataset_manifest_path='manifest.json',
 parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[])
evaluation=dict(cpu_threads=4,strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,bootstrap_resamples=200,
 bootstrap_block_points=1024,tab_threshold_ratios_percent=[.1,.5,1,2,3,5,10,15,20,25],tab_commit='fixture')
torch.set_num_threads(4);source.run(root,job,evaluation)
result=s.read(root/'source/result.json');binding=dict(result_sha256=s.sha(root/'source/result.json'))
s.check_source=lambda runtime,case:(result,dataset,binding)
s.write(root/'runtime.json',dict(fixture_only=True))
runtime=dict(artifact_root=str(root/'replay'),_config_path='runtime.json',math_tolerance=dict(rtol=2e-6,atol=2e-6),existing_native_certificate=dict(id='not_this_fixture'))
case=dict(job=job,config=config,queue=dict(evaluation=evaluation))
s.native_worker(runtime,case);s.review_worker(runtime,case)
native=s.read(root/'replay/fixture/native/native_replay.json');review=s.read(root/'replay/fixture/independent/review.json')
assert native['passed'] and review['passed'] and native['fit_calls']==0 and native['training_updates_performed']==0
assert len(native['full_split_coverage'])==6
obs=native['full_MC_probability_path_and_score_formula_observation'];assert obs['path_calls']==obs['forward_calls']==120
assert obs['windows']==16 and obs['score_calls']==6 and obs['window_points']==1600
assert review['inline_path_formula_audited_by_new_native_stage'] and not review['original_author_TSAD_equivalence']
assert obs['source_FP32_path_program_checked']
if objective=='target':
 assert obs['float64_path_failure_coordinate_counts']['velocity']>0
 assert not native['ideal_float64_path_equivalence_certified'] and not review['ideal_float64_path_equivalence_certified']
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(subject.ROOT)+os.pathsep+str(subject.ROOT/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(tmp_path),objective,str(weight),str(stable)],cwd=subject.ROOT,env=env,capture_output=True,text=True,timeout=150)
    assert p.returncode==0,p.stdout[-1500:]+p.stderr[-5000:]
