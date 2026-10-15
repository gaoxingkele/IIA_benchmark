"""Full original local capacity/epochs on tiny CPU fixtures,not a benchmark."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import replay_full5_DCD105_min105_controls_v1 as s


def test_all5_full54entity_sources_and20labels_retain_both_original_resource_policies():
    c=s.registration();cases=s.verify(c)
    assert len(cases)==5 and sum(len(x['required_label_policies']) for x in cases)==20
    assert {x['seed'] for x in cases}==set(range(1103,1108))
    native=s.read(s.ROOT/s.baseline.POLICY);scalar=s.read(s.ROOT/s.metric.POLICY)
    assert c['settings']==native['settings'] and c['resource_lock']==native['resource_lock']
    assert c['scalar_settings']==scalar['settings'] and c['scalar_resource_lock']==scalar['resource_lock']
    assert c['math_tolerance']==dict(atol=2e-6,rtol=2e-6)
    for case in cases:
        q,j=s.metric.resolve_case(case);m=s.read(s.ROOT/j['model_config'])
        s.baseline.validate_model(c,case,q,j,m)
        assert m['parameters']['epochs']==3 and m['parameters']['parameters']['window']==105
        assert len(s.producer.binding(s.read(s.ROOT/s.producer.CONFIG))['records']['smap']['entities'])==54


@pytest.mark.parametrize('fault',['case','labels','native_memory','scalar_memory','native_lock','scalar_lock','math','source','models'])
def test_wrong_scope_source_or_weakened_policy_rejected(fault):
    c=s.registration()
    if fault=='case':c['cases'].pop()
    elif fault=='labels':c['cases'][0]['required_label_policies'].pop()
    elif fault=='native_memory':c['settings']['minimum_commit_headroom_bytes']=28*2**30
    elif fault=='scalar_memory':c['scalar_settings']['minimum_commit_headroom_bytes']=1
    elif fault=='native_lock':c['resource_lock']=c['scalar_resource_lock']
    elif fault=='scalar_lock':c['scalar_resource_lock']='new_lock'
    elif fault=='math':c['math_tolerance']['atol']=1e-3
    elif fault=='source':c['source_execution']=s.metric.CONFIG
    else:c['source_model_configurations'].clear()
    with pytest.raises(ValueError):s.verify(c)


def test_sources_still_require_actual_successful_source_guard():
    c=s.registration()
    for case in c['cases']:
        if s.metric.source_status(case) is None:
            with pytest.raises(ValueError,match='completed source'):s.source_binding(c,case)


def test_phase_commands_have_exact_original_interpreter_and_unambiguous_phase():
    case=s.registration()['cases'][0]
    for phase in s.REPORTS:
        command=s.worker_command(case,phase)
        assert command[-2:]==['--stage',phase] and command[command.index('--job-id')+1]==case['id']
        assert command[0]==str(s.ROOT/case['worker_python'])
    with pytest.raises(ValueError):s.worker_command(case,'train')


@pytest.mark.parametrize('fault',['case','config','guard_exit','guard_abort','guard_hash','guard_command','labels','arrays','thresholds','binding'])
def test_scalar_proof_guard_and_every_original_label_view_are_required(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'CONFIG','runtime.json')
    s.write(tmp_path/'runtime.json',dict(fixture_only=True))
    c=dict(state_root='state',artifact_root=str(tmp_path/'artifacts'))
    case=dict(id='fixture',worker_python=sys.executable,required_label_policies=['first','last'])
    attempt,output=s.paths(c,case,'scalar');attempt.mkdir(parents=True);output.mkdir(parents=True)
    proof=dict(passed=True,case=deepcopy(case),configuration_sha256=s.sha(tmp_path/'runtime.json'),
        source_metrics_exactly_recomputed=True,all_original_arrays_checked=True,
        label_view_results=dict(first={},last={}),all_validation_thresholds_and_predictions_unchanged=True,
        source_binding=dict(fixture_only=True))
    if fault=='case':proof['case']['id']='other'
    elif fault=='config':proof['configuration_sha256']='changed'
    elif fault=='labels':proof['label_view_results'].pop('last')
    elif fault=='arrays':proof['all_original_arrays_checked']=False
    elif fault=='thresholds':proof['all_validation_thresholds_and_predictions_unchanged']=False
    elif fault=='binding':proof['source_binding']={}
    target=output/s.REPORTS['scalar'];s.write(target,proof)
    guard=dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=s.sha(target),
        exact_child_handle=dict(command=s.worker_command(case,'scalar')))
    if fault=='guard_exit':guard['exit_code']=1
    elif fault=='guard_abort':guard['resource_guard_aborted']=True
    elif fault=='guard_hash':guard['proof_sha256']='changed'
    elif fault=='guard_command':guard['exact_child_handle']['command'][-1]='review'
    s.write(attempt/'guard_receipt.json',guard)
    monkeypatch.setattr(s,'source_binding',lambda c,case:({},None,None,None,dict(fixture_only=True)))
    with pytest.raises(ValueError):s.ready_scalar(c,case)


FIXTURE=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_full5_DCD105_min105_controls_v1 as s
from scripts.flow_matching import run_tsad_experiment as source
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];s.ROOT=root;s.CONFIG='fixture_runtime.json'
m=s.read(root/'fixture_metadata.json');c=m['runtime'];case=m['case'];job=m['job'];dataset=m['dataset'];evaluation=m['evaluation']
if phase=='scalar':
 rng=np.random.default_rng(1103);arrays={}
 for entity in dataset['entities']:
  prefix=entity['array_prefix']
  for role,n in [('train',107),('validation',105),('test',217)]:arrays[prefix+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
  labels=np.zeros(217,np.int8);labels[50:70]=1;arrays[prefix+'_labels']=labels
 np.savez_compressed(root/'inputs.npz',**arrays);dataset['input_sha256']=s.sha(root/'inputs.npz')
 s.write(root/'manifest.json',dict(datasets=[dataset]));s.write(root/'model.json',m['model']);s.write(root/'fixture_runtime.json',c)
 for index,policy in enumerate(dataset['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in dataset['entities']}
  if index:labels['a_labels'][70:70+index]=1
  path=root/(policy+'.npz');np.savez_compressed(path,**labels)
  dataset['labels_only_controls'][policy].update(path=str(path),sha256=s.sha(path))
 torch.set_num_threads(2);source.run(root,job,evaluation)
 binding=dict(fixture_only=True,source_result_sha256=s.sha(root/'source/result.json'))
 m.update(dataset=dataset,binding=binding);(root/'fixture_metadata.json').write_text(json.dumps(m))
else:binding=m['binding']
result=s.read(root/'source/result.json')
s.source_binding=lambda config,case:(dict(evaluation=evaluation),job,result,dataset,binding)
# Fixture providers are confined to this child. No production guard is emitted.
_,scalar_dir=s.paths(c,case,'scalar');scalar_path=scalar_dir/s.REPORTS['scalar']
if phase!='scalar':
 s.ready_scalar=lambda config,case:dict(path=str(scalar_path),sha256=s.sha(scalar_path),fixture_only=True)
if phase=='review':
 _,native_dir=s.paths(c,case,'native');native_path=native_dir/s.REPORTS['native']
 s.ready_native=lambda config,case:(native_path,s.read(native_path))
dict(scalar=s.scalar_worker,native=s.native_worker,review=s.review_worker)[phase](c,case)
_,output=s.paths(c,case,phase);proof=s.read(output/s.REPORTS[phase]);assert proof['passed']
if phase=='scalar':
 assert len(proof['label_view_results'])==4 and proof['all_original_arrays_checked']
 assert not proof['training_or_model_inference_performed']
elif phase=='native':
 assert proof['all_array_dtype_shape_content_bytes_identical'] and proof['all_model_tensors_identical_before_and_after']
 assert proof['additional_fit_calls']==proof['additional_optimizer_updates']==0
 assert proof['all_source_forward_and_FP64_formula_observation']['windows']==12
 assert sum(r['points'] for r in proof['full_native_coverage'])==858
 assert proof['math_tolerance']==dict(atol=2e-6,rtol=2e-6)
else:
 assert proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
 assert len(proof['all_label_view_results'])==4 and not proof['new_training_or_model_inference_in_review']
'''


def test_full_local3epoch_256width_threepatch_model_reloads_and_reviews_short_real_geometries(tmp_path):
    c=s.registration();actual=c['cases'][0];queue,old=s.metric.resolve_case(actual)
    model=s.read(s.ROOT/old['model_config'])
    case=dict(id='fixture_only_not_benchmark',family='baselines',seed=1103,window=105,worker_python=sys.executable)
    job=dict(id=case['id'],dataset='fixture',seed=1103,window=105,model_config='model.json',dataset_manifest_path='manifest.json',
        parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[])
    dataset=dict(id='fixture',input_npz='inputs.npz',primary_label_policy='source_first_exclusive',grouped_splits_verified=True,
        entities=[dict(entity_id='a',array_prefix='a',features=2),dict(entity_id='b',array_prefix='b',features=2)],
        labels_only_controls={p:{} for p in actual['required_label_policies']})
    runtime=dict(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),math_tolerance=c['math_tolerance'],
        expected_source_model_entrypoint=c['expected_source_model_entrypoint'],source_model_configurations={'model.json':model})
    s.write(tmp_path/'fixture_metadata.json',dict(runtime=runtime,case=case,job=job,dataset=dataset,model=model,evaluation=queue['evaluation'],
        fixture_only_not_benchmark=True,actual_local_source_training_budget_preserved=True))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for phase in ['scalar','native','review']:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-c',FIXTURE,str(tmp_path),phase],cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=240)
        assert p.returncode==0,p.stdout[-1500:]+p.stderr[-5000:]


def test_complete_replay_registration_is_cold_without_model_or_data_array_imports():
    code='from scripts.flow_matching import replay_full5_DCD105_min105_controls_v1 as s;import sys;s.verify(s.registration());assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
