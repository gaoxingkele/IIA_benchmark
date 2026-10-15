"""Full-budget tiny-source checkpoints test plumbing,not benchmark performance."""
from copy import deepcopy
from collections import Counter
import json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.flow_matching import replay_original_full105_baselines_v1 as s


def select_config(method='anomaly_transformer',dataset='smap',adversarial=None):
    c=s.registration()
    for case in c['cases']:
        if case['dataset']!=dataset or case['seed']!=1103:continue
        q,j=s.scalar.resolve_case(case);m=s.read(s.ROOT/j['model_config']);p=m['parameters']
        if p['method']==method and p['parameters'].get('adversarial')==adversarial:return c,deepcopy(case),deepcopy(q),deepcopy(j),deepcopy(m)
    raise AssertionError('Original baseline variant absent')


def test_all105_original_cases245labels_all21configs_and_full_resource_scope():
    c=s.registration();cases=s.verify(c);old=s.read(s.ROOT/s.scalar.CONFIG)
    assert cases==[x for x in old['cases'] if x['family']=='baselines']
    assert len(cases)==105 and sum(len(x['required_label_policies']) for x in cases)==245
    assert Counter(x['dataset'] for x in cases)==dict(smap=35,msl=35,smd=35)
    assert len(c['source_model_configurations'])==21
    assert c['original_scope']==old['original_scope'] and c['original_all_family_cases']==873 and c['original_all_label_views']==2037
    policy=s.read(s.ROOT/s.POLICY)
    assert c['settings']==policy['settings'] and c['resource_lock']==policy['resource_lock']
    assert c['math_tolerance']==dict(atol=2e-6,rtol=2e-6)
    for case in cases:
        q,j=s.scalar.resolve_case(case);m=s.read(s.ROOT/j['model_config']);s.validate_model(c,case,q,j,m)


@pytest.mark.parametrize('fault',['cases','required_family_native_cases','required_family_label_views','math_tolerance','settings','resource_lock','source_model_configurations'])
def test_weaker_scope_math_or_resource_contract_rejected(fault):
    c=s.registration()
    if fault=='cases':c[fault].pop()
    elif fault=='math_tolerance':c[fault]['atol']=.001
    elif fault=='settings':c[fault]['minimum_commit_headroom_bytes']=28*2**30
    elif fault=='resource_lock':c[fault]='different_lane'
    elif fault=='source_model_configurations':c[fault].pop(next(iter(c[fault])))
    else:c[fault]=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('fault',['epochs','width','window','batch','patch','loss_sign','seed','device','threads','entrypoint'])
def test_original_dataset_architecture_patch_loss_sign_and_training_budget_required(fault):
    method='usad' if fault=='loss_sign' else 'dcdetector';adv=False if fault=='loss_sign' else None
    c,case,q,j,m=select_config(method,adversarial=adv);p=m['parameters']
    if fault=='epochs':p['epochs']=1
    elif fault=='width':p['parameters']['d_model']=16
    elif fault=='window':case['window']=10
    elif fault=='batch':p['parameters']['batch_size']=1
    elif fault=='patch':p['parameters']['patch_sizes']=[3]
    elif fault=='loss_sign':p['parameters']['adversarial']=True
    elif fault=='seed':j['parameter_overrides']['seed']=42
    elif fault=='device':j['parameter_overrides']['device']='cuda'
    elif fault=='threads':q['evaluation']['cpu_threads']=4
    else:m['entrypoint']='replacement'
    with pytest.raises(ValueError):s.validate_model(c,case,q,j,m)


@pytest.mark.parametrize('losses',[[],[1.]*20])
def test_never_invent_unpersisted_legacy_training_history(losses):
    with pytest.raises(ValueError):s.validate_training_result(dict(training_losses=losses,parameter_count=100))
    s.validate_training_result(dict(training_losses=None,parameter_count=100))


def observation():
    model=dict(parameters=dict(method='anomaly_transformer',parameters=dict(window=100,batch_size=256,n_heads=8,e_layers=3)))
    coverage=[dict(windows=3,features=2),dict(windows=3,features=2)]
    report=dict(method='anomaly_transformer',source_score_batch_size=256,windows=6,window_points=600,
        forward_calls=2,score_calls=2,reconstruction_coordinates_checked=1200,
        attention_probability_coordinates_checked=2880000,per_head_KL_calls_checked=12,
        source_FP32_energy_program_checked=True,independent_FP64_score_program_checked=True,
        independent_neural_kernel_implementation=False,ideal_FP64_score_failure_coordinates=5,
        ideal_FP64_complete_score_equivalence_certified=False)
    return coverage,report,model


def test_ideal_FP64_mismatch_is_retained_without_false_equivalence_claim():
    coverage,report,model=observation();s.validate_observation(coverage,report,model)
    assert not report['ideal_FP64_complete_score_equivalence_certified']


@pytest.mark.parametrize('fault',['windows','window_points','forward_calls','score_calls','reconstruction_coordinates_checked',
    'attention_probability_coordinates_checked','per_head_KL_calls_checked','source_FP32_energy_program_checked',
    'ideal_FP64_complete_score_equivalence_certified'])
def test_missing_inputs_heads_or_KL_calls_or_hidden_FP64_failures_rejected(fault):
    coverage,report,model=observation()
    if fault=='source_FP32_energy_program_checked':report[fault]=False
    elif fault=='ideal_FP64_complete_score_equivalence_certified':report[fault]=True
    else:report[fault]-=1
    with pytest.raises(ValueError):s.validate_observation(coverage,report,model)


FIXTURE_SCRIPT=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_original_full105_baselines_v1 as s
from scripts.flow_matching import run_tsad_experiment as source
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];s.ROOT=root;s.CONFIG='fixture_runtime.json'
metadata=s.read(root/'fixture_metadata.json');model=metadata['model'];evaluation=metadata['evaluation']
case=metadata['case'];job=metadata['job'];dataset=metadata['dataset'];config=metadata['runtime']
if phase=='native':
 rng=np.random.default_rng(1103);arrays={}
 for entity in dataset['entities']:
  prefix=entity['array_prefix']
  for role,n in [('train',2*job['window']+1),('validation',job['window']+5),('test',2*job['window']+5)]:arrays[prefix+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
  labels=np.zeros(2*job['window']+5,np.int8);labels[20:50]=1;arrays[prefix+'_labels']=labels
 np.savez_compressed(root/'fixture_inputs.npz',**arrays);dataset['input_sha256']=s.sha(root/'fixture_inputs.npz')
 s.write(root/'fixture_manifest.json',dict(datasets=[dataset]));s.write(root/'fixture_model.json',model)
 for n,policy in enumerate(dataset['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in dataset['entities']}
  if n:labels['a_labels'][-1-n]=1
  target=root/(policy+'.npz');np.savez_compressed(target,**labels)
  dataset['labels_only_controls'][policy].update(path=str(target),sha256=s.sha(target))
 s.write(root/'fixture_runtime.json',config);torch.set_num_threads(2)
 source.run(root,job,evaluation)
 result=s.read(root/'source/result.json');binding=dict(fixture_only=True,result_sha256=s.sha(root/'source/result.json'))
 with np.load(root/'fixture_inputs.npz',allow_pickle=False) as inputs,np.load(root/'source/scores.npz',allow_pickle=False) as saved:
  entities,val,train,coverage,digests=s.scalar.collect_arrays(inputs,saved,dataset)
 overlays={}
 for p,label in dataset['labels_only_controls'].items():
  with np.load(label['path'],allow_pickle=False) as saved:overlays[p]={k:saved[k] for k in saved.files}
 views=s.scalar.evaluate_views(entities,val,train,overlays,dataset,result,evaluation,1103)
 s.write(root/'fixture_scalar.json',dict(label_view_results=views,source_array_digests=digests,full_positional_saved_score_coverage=coverage))
 metadata.update(dataset=dataset,binding=binding);(root/'fixture_metadata.json').write_text(json.dumps(metadata))
else:result=s.read(root/'source/result.json');binding=metadata['binding']
# Dedicated unit-process providers isolate numerical bodies from production
# source gates. No fake scientific guard or real benchmark receipt is emitted.
scalar_evidence=dict(path=str(root/'fixture_scalar.json'),sha256=s.sha(root/'fixture_scalar.json'),fixture_only=True)
s.scalar.source_binding=lambda config,case:(dict(evaluation=evaluation),job,result,dataset,binding)
s.ready_scalar=lambda config,case:scalar_evidence
if phase=='native':
 s.native_worker(config,case)
else:
 _,native_dir=s.paths(config,case,'native');native_path=native_dir/s.REPORTS['native']
 s.ready_native=lambda config,case:(native_path,s.read(native_path))
 s.review_worker(config,case)
_,output=s.paths(config,case,phase);proof=s.read(output/s.REPORTS[phase]);assert proof['passed']
if phase=='native':
 obs=proof['all_source_forward_and_FP64_formula_observation']
 assert obs['windows']==16 and obs['score_calls']==6 and obs['window_points']==16*job['window']
 assert obs['forward_calls']==(18 if model['parameters']['method']=='usad' else 6)
 assert proof['additional_fit_calls']==proof['additional_optimizer_updates']==0
 assert proof['all_array_dtype_shape_content_bytes_identical'] and proof['all_model_tensors_identical_before_and_after']
 s.validate_observation(proof['full_native_coverage'],obs,model)
 assert proof['independent_FP64_scoring_formula_verified'] and proof['math_tolerance']==dict(rtol=2e-6,atol=2e-6)
else:
 assert len(proof['all_label_view_results'])==4
 assert proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
 assert proof['all_original_and_native_arrays_independently_compared']
'''


@pytest.mark.parametrize('method,dataset,adversarial',[
    ('anomaly_transformer','smap',None),('dcdetector','smap',None),('dcdetector','msl',None),('dcdetector','smd',None),
    ('itransformer','smap',None),('timesnet','smap',None),('timesnet','msl',None),('timesnet','smd',None),
    ('tranad','smap',None),('usad','smap',False),('usad','smap',True)])
def test_original_budget_source_complete_checkpoint_all_forward_math_and_fresh_review(tmp_path,method,dataset,adversarial):
    c,actual,queue,old,model=select_config(method,dataset,adversarial)
    case=dict(id='synthetic_fixture_only_not_benchmark',family='baselines',seed=1103,window=old['window'],worker_python=sys.executable)
    job=dict(id=case['id'],dataset='fixture',seed=1103,window=old['window'],model_config='fixture_model.json',dataset_manifest_path='fixture_manifest.json',
        parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[])
    data=dict(id='fixture',input_npz='fixture_inputs.npz',grouped_splits_verified=True,primary_label_policy='source_first_exclusive',
        entities=[dict(entity_id='a',array_prefix='a',features=2),dict(entity_id='b',array_prefix='b',features=2)],
        labels_only_controls={p:{} for p in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']})
    runtime=dict(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),expected_source_model_entrypoint=c['expected_source_model_entrypoint'],
        source_model_configurations={'fixture_model.json':model},math_tolerance=c['math_tolerance'])
    metadata=dict(model=model,case=case,job=job,dataset=data,evaluation=queue['evaluation'],runtime=runtime,
        fixture_only_no_benchmark_performance=True)
    (tmp_path/'fixture_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for phase in ['native','review']:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',FIXTURE_SCRIPT,str(tmp_path),phase],cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=240)
        assert result.returncode==0,result.stdout[-1000:]+result.stderr[-5000:]


def test_cold_complete_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import replay_original_full105_baselines_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
