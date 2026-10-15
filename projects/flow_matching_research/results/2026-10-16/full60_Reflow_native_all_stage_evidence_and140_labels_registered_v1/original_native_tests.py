"""Full-budget tiny numerical fixtures are plumbing,never benchmark performance."""
from copy import deepcopy
from collections import Counter
import json,os
from pathlib import Path
import subprocess,sys
import pytest
from scripts.flow_matching import replay_original_full60_reflow_v1 as s


def select_config(variant='fm_reflow_2stage'):
    c=s.registration()
    for case in c['cases']:
        if case['dataset']!='smap' or case['seed']!=1103:continue
        q,j=s.scalar.resolve_case(case);m=s.read(s.ROOT/j['model_config'])
        if m['id']==variant:return c,deepcopy(case),deepcopy(q),deepcopy(j),deepcopy(m)
    raise AssertionError('All four original variants required')


def test_all60_cases140views_fullscope_and_original_resource_contract():
    c=s.registration();cases=s.verify(c);old=s.read(s.ROOT/s.scalar.CONFIG)
    assert cases==[x for x in old['cases'] if x['family']=='reflow']
    assert len(cases)==60 and sum(len(x['required_label_policies']) for x in cases)==140
    assert Counter(x['dataset'] for x in cases)==dict(smap=20,msl=20,smd=20)
    p=s.read(s.ROOT/s.POLICY)
    assert c['settings']==p['settings'] and c['resource_lock']==p['resource_lock']
    assert c['math_tolerance']==dict(atol=2e-6,rtol=2e-6)
    assert c['original_scope']==old['original_scope'] and c['original_all_family_cases']==873
    assert c['original_all_label_views']==2037 and c['other_family_cases']==813


@pytest.mark.parametrize('fault',['cases','required_family_native_cases','required_family_label_views','math_tolerance','settings','resource_lock'])
def test_weaker_scope_or_resource_contract_rejected(fault):
    c=s.registration()
    if fault=='cases':c[fault].pop()
    elif fault=='math_tolerance':c[fault]['atol']=.001
    elif fault=='settings':c[fault]['minimum_commit_headroom_bytes']=28*2**30
    elif fault=='resource_lock':c[fault]='different_lane'
    else:c[fault]=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('fault',['epochs','hidden_size','window','batch_size','monte_carlo_samples','score_times','learning_rate','reflow_stages','reflow_epochs','reflow_ode_steps','reflow_solver','threads','total','paircompute','seed','device'])
def test_complete_original_source_budget_and_allstage_definitions_required(fault):
    c,case,q,j,m=select_config();p=m['parameters']
    if fault=='threads':q['evaluation']['cpu_threads']=4
    elif fault=='total':j['total_training_epochs']=20
    elif fault=='paircompute':j['additional_pair_generation_compute']=False
    elif fault=='seed':j['parameter_overrides']['seed']=42
    elif fault=='device':j['parameter_overrides']['device']='cuda'
    elif fault=='score_times':p[fault]=[.5]
    elif fault=='reflow_solver':p[fault]='euler'
    else:p[fault]=1
    with pytest.raises(ValueError):s.validate_model(c,case,q,j,m)


@pytest.mark.parametrize('variant',['fm_reflow_2stage','fm_reflow_3stage','fm_rectified_40epoch_budget_control','fm_rectified_60epoch_budget_control'])
def test_every_registered_variant_preserves_its_budget_and_extra_compute_declaration(variant):
    c,case,q,j,m=select_config(variant);s.validate_model(c,case,q,j,m)
    assert j['total_training_epochs'] in [40,60]
    assert j['additional_pair_generation_compute']==('reflow_stages' in m['parameters'])


@pytest.mark.parametrize('losses',[None,[],[1.]*39,[1.]*39+[float('nan')],[1.]*39+[-1.]])
def test_full_finite_source_history_required(losses):
    with pytest.raises(ValueError):s.validate_training_result(dict(training_losses=losses,parameter_count=100),dict(total_training_epochs=40))


FIXTURE_SCRIPT=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_original_full60_reflow_v1 as s
from scripts.flow_matching import run_tsad_experiment as source
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];s.ROOT=root;s.CONFIG='fixture_runtime.json'
metadata=s.read(root/'fixture_metadata.json');model=metadata['model'];evaluation=metadata['evaluation']
case=metadata['case'];job=metadata['job'];dataset=metadata['dataset'];config=metadata['runtime']
if phase=='native':
 rng=np.random.default_rng(1103);arrays={}
 for entity in dataset['entities']:
  prefix=entity['array_prefix']
  for role,n in [('train',201),('validation',105),('test',205)]:arrays[prefix+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
  labels=np.zeros(205,np.int8);labels[20:50]=1;arrays[prefix+'_labels']=labels
 np.savez_compressed(root/'fixture_inputs.npz',**arrays);dataset['input_sha256']=s.sha(root/'fixture_inputs.npz')
 s.write(root/'fixture_manifest.json',dict(datasets=[dataset]));s.write(root/'fixture_model.json',model)
 for n,policy in enumerate(dataset['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in dataset['entities']}
  if n:labels['a_labels'][50+n]=1
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
 obs=proof['full_MC_probability_path_and_score_formula_observation']
 assert obs['windows']==16 and obs['score_calls']==6 and obs['window_points']==1600
 assert obs['path_calls']==obs['forward_calls']==120 and proof['additional_fit_calls']==proof['additional_optimizer_updates']==0
 assert proof['all_array_dtype_shape_content_bytes_identical'] and proof['all_model_tensors_identical_before_and_after']
 s.validate_coupling_summary(proof)
 assert len(proof['all_stage_coupling_evidence']['stages'])==model['parameters'].get('reflow_stages',1)-1
 assert proof['source_FP32_operation_program_verified'] and proof['math_tolerance']==dict(rtol=2e-6,atol=2e-6)
else:
 assert len(proof['all_label_view_results'])==4
 assert proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
 assert proof['all_original_and_native_arrays_independently_compared']
'''


@pytest.mark.parametrize('variant',['fm_reflow_2stage','fm_reflow_3stage','fm_rectified_40epoch_budget_control','fm_rectified_60epoch_budget_control'])
def test_full40or60epoch_source_all_saved_stage_pairs_native_and_fresh_review(tmp_path,variant):
    c,actual,queue,old,model=select_config(variant)
    case=dict(id='synthetic_fixture_only_not_benchmark',family='reflow',seed=1103,window=100,worker_python=sys.executable)
    job=dict(id=case['id'],dataset='fixture',seed=1103,window=100,model_config='fixture_model.json',dataset_manifest_path='fixture_manifest.json',
        parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[],
        total_training_epochs=old['total_training_epochs'],additional_pair_generation_compute=old['additional_pair_generation_compute'])
    data=dict(id='fixture',input_npz='fixture_inputs.npz',grouped_splits_verified=True,primary_label_policy='source_first_exclusive',
        entities=[dict(entity_id='a',array_prefix='a',features=2),dict(entity_id='b',array_prefix='b',features=2)],
        labels_only_controls={p:{} for p in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']})
    runtime=dict(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),expected_source_model_entrypoints=c['expected_source_model_entrypoints'],math_tolerance=c['math_tolerance'])
    metadata=dict(model=model,case=case,job=job,dataset=data,evaluation=queue['evaluation'],runtime=runtime,
        fixture_only_no_benchmark_performance=True)
    (tmp_path/'fixture_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for phase in ['native','review']:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',FIXTURE_SCRIPT,str(tmp_path),phase],cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout[-1000:]+result.stderr[-5000:]


def test_cold_complete_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import replay_original_full60_reflow_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
