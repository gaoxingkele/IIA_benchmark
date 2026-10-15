"""All34original budgets on tiny CPUfixtures are plumbing,not CUDA/benchmark evidence."""
from copy import deepcopy
from collections import Counter
import json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.flow_matching import replay_original_full510_neural_v1 as s


def select_config(identity='fm_comparator_dt_la'):
    c=s.registration()
    for case in c['cases']:
        if case['dataset']!='smap' or case['seed']!=1103:continue
        q,j=s.scalar.resolve_case(case);m=s.read(s.ROOT/j['model_config'])
        if m['id']==identity:return c,deepcopy(case),deepcopy(q),deepcopy(j),deepcopy(m)
    raise AssertionError('Every full original variant must remain')


def test_every510_case1190labels34models_and_both_source_device_resource_policies():
    c=s.registration();cases=s.verify(c);old=s.read(s.ROOT/s.scalar.CONFIG)
    assert cases==[x for x in old['cases'] if x['family']=='neural']
    assert len(cases)==510 and sum(len(x['required_label_policies']) for x in cases)==1190
    assert Counter(x['dataset'] for x in cases)==dict(smap=170,msl=170,smd=170)
    assert len(c['source_model_configurations'])==34
    assert c['original_scope']==old['original_scope'] and c['original_all_family_cases']==873 and c['original_all_label_views']==2037
    cpu,gpu=s.read(s.ROOT/s.POLICY),s.read(s.ROOT/s.GPU_POLICY)
    assert c['settings']==cpu['settings'] and c['resource_lock']==cpu['resource_lock']
    assert c['GPU_settings']==gpu['settings'] and c['GPU_resource_lock']==gpu['resource_lock']
    assert c['expected_source_device']=='cuda:0' and c['source_cpu_threads']==4
    assert c['math_tolerance']==dict(atol=2e-6,rtol=2e-6)
    for case in cases:
        q,j=s.scalar.resolve_case(case);m=s.read(s.ROOT/j['model_config']);s.validate_model(c,case,q,j,m)


@pytest.mark.parametrize('fault',['cases','required_family_native_cases','required_family_label_views','math_tolerance','settings',
    'GPU_settings','resource_lock','GPU_resource_lock','gpu_prerequisite_queues','legacy_gpu_queue_lock','expected_source_device'])
def test_original_scope_CUDA_precision_and_phase_resource_contract_cannot_be_weakened(fault):
    c=s.registration()
    if fault=='cases':c[fault].pop()
    elif fault=='math_tolerance':c[fault]['atol']=.001
    elif fault=='settings':c[fault]['minimum_commit_headroom_bytes']=28*2**30
    elif fault=='GPU_settings':c[fault]['minimum_gpu_free_bytes']=4*2**30
    elif fault=='gpu_prerequisite_queues':c[fault]=[]
    elif fault in ['resource_lock','GPU_resource_lock','legacy_gpu_queue_lock']:c[fault]='different_lane'
    elif fault=='expected_source_device':c[fault]='cpu'
    else:c[fault]=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('identity,fault',[
    ('fm_comparator_dt_la','epochs'),('fm_comparator_dt_la','width'),('fm_comparator_dt_la','score_mode'),
    ('fm_comparator_shcl_transformer','branches'),('fm_comparator_kgl','use_kan'),('fm_native_dfm_exact','trace'),
    ('fm_native_grasp_grasp','source_samples'),('fm_native_grasp_grasp','flow_evaluations'),
    ('fm_comparator_dt_la','device'),('fm_comparator_dt_la','seed'),('fm_comparator_dt_la','threads')])
def test_every_original_capacity_epoch_branch_MC_trace_seed_and_numerical_device_required(identity,fault):
    c,case,q,j,m=select_config(identity);p=m['parameters']
    if fault=='epochs':p[fault]=1
    elif fault=='width':p['options']['d_model']=16
    elif fault=='score_mode':p['options'][fault]='input'
    elif fault=='branches':p['options'][fault]=['point']
    elif fault=='use_kan':p['options'][fault]=False
    elif fault=='trace':p[fault]='hutchinson'
    elif fault in ['source_samples','flow_evaluations']:p[fault]=1
    elif fault=='device':j['parameter_overrides']['device']='cpu'
    elif fault=='seed':j['parameter_overrides']['seed']=42
    else:q['evaluation']['cpu_threads']=2
    with pytest.raises(ValueError):s.validate_model(c,case,q,j,m)


FIXTURE_SCRIPT=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_original_full510_neural_v1 as s
from scripts.flow_matching import run_tsad_experiment as source
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];s.ROOT=root;s.CONFIG='fixture_runtime.json'
metadata=s.read(root/'fixture_metadata.json');model=metadata['model'];evaluation=metadata['evaluation']
case=metadata['case'];job=metadata['job'];dataset=metadata['dataset'];config=metadata['runtime']
if phase=='native':
 rng=np.random.default_rng(1103);arrays={}
 for entity in dataset['entities']:
  prefix=entity['array_prefix']
  for role,n in [('train',job['window']+1),('validation',job['window']+3),('test',2*job['window']+3)]:arrays[prefix+'_'+role]=rng.normal(size=(n,3)).astype(np.float32)
  arrays[prefix+'_train'][:,1]=arrays[prefix+'_train'][:,0]+.05*arrays[prefix+'_train'][:,1]
  labels=np.zeros(2*job['window']+3,np.int8);labels[2:6]=1;arrays[prefix+'_labels']=labels
 np.savez_compressed(root/'fixture_inputs.npz',**arrays);dataset['input_sha256']=s.sha(root/'fixture_inputs.npz')
 s.write(root/'fixture_manifest.json',dict(datasets=[dataset]));s.write(root/'fixture_model.json',model)
 for n,policy in enumerate(dataset['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in dataset['entities']}
  if n:labels['a_labels'][-1-n]=1
  target=root/(policy+'.npz');np.savez_compressed(target,**labels)
  dataset['labels_only_controls'][policy].update(path=str(target),sha256=s.sha(target))
 s.write(root/'fixture_runtime.json',config);torch.set_num_threads(4)
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
 assert obs['windows']==14 and obs['score_calls']==6 and obs['window_points']==14*job['window']
 assert proof['additional_fit_calls']==proof['additional_optimizer_updates']==0
 assert proof['all_array_dtype_shape_content_bytes_identical'] and proof['all_model_tensors_identical_before_and_after']
 s.validate_observation(proof['full_native_coverage'],obs,model)
 assert proof['independent_FP64_scoring_formula_verified'] and proof['math_tolerance']==dict(rtol=2e-6,atol=2e-6)
else:
 assert len(proof['all_label_view_results'])==4
 assert proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
 assert proof['all_original_and_native_arrays_independently_compared']
'''


IDS=sorted(model['id'] for model in s.registration()['source_model_configurations'].values())
assert len(IDS)==34


@pytest.mark.parametrize('identity',IDS)
def test_full_original_epoch_capacity_and_each_ablation_checkpoint_on_tiny_CPU_fixture(tmp_path,identity):
    c,actual,queue,old,model=select_config(identity)
    case=dict(id='synthetic_CPU_fixture_only_not_CUDA_or_benchmark',family='neural',seed=1103,window=old['window'],worker_python=sys.executable)
    job=dict(id=case['id'],dataset='fixture',seed=1103,window=old['window'],model_config='fixture_model.json',dataset_manifest_path='fixture_manifest.json',
        parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[])
    data=dict(id='fixture',input_npz='fixture_inputs.npz',grouped_splits_verified=True,primary_label_policy='source_first_exclusive',
        entities=[dict(entity_id='a',array_prefix='a',features=3),dict(entity_id='b',array_prefix='b',features=3)],
        labels_only_controls={p:{} for p in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']})
    runtime=dict(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),
        expected_source_model_entrypoints=c['expected_source_model_entrypoints'],expected_source_device='cpu',source_cpu_threads=4,
        source_model_configurations={'fixture_model.json':model},math_tolerance=c['math_tolerance'])
    metadata=dict(model=model,case=case,job=job,dataset=data,evaluation=queue['evaluation'],runtime=runtime,
        fixture_only_no_CUDA_or_benchmark_performance=True)
    (tmp_path/'fixture_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for phase in ['native','review']:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',FIXTURE_SCRIPT,str(tmp_path),phase],cwd=s.ROOT,env=env,
            capture_output=True,text=True,timeout=900)
        assert result.returncode==0,result.stdout[-1200:]+result.stderr[-6000:]


def test_cold_complete_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import replay_original_full510_neural_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
