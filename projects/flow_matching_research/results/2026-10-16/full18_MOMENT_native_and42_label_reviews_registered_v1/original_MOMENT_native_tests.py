"""Real Small release weights on tiny inputs test plumbing,not benchmarks."""
from copy import deepcopy
from collections import Counter
import json,os
from pathlib import Path
import subprocess,sys
import pytest
from scripts.flow_matching import replay_original_full18_moment_v1 as s


def select(window=100):
    c=s.registration()
    for case in c['cases']:
        if case['window']==window and case['dataset']=='smap' and '_small__' in case['id']:
            q,j=s.scalar.resolve_case(case);return c,deepcopy(case),deepcopy(q),deepcopy(j),s.read(s.ROOT/j['model_config'])
    raise AssertionError('Registered Small context missing')


def test_all18_sizes_contexts_datasets_all42views_and_full_source_scope():
    c=s.registration();cases=s.verify(c)
    assert len(cases)==18 and sum(len(x['required_label_policies']) for x in cases)==42
    assert Counter(x['dataset'] for x in cases)==dict(smap=6,msl=6,smd=6)
    assert Counter(x['window'] for x in cases)=={100:9,512:9} and {x['seed'] for x in cases}=={42}
    policy=s.read(s.ROOT/s.POLICY)
    assert c['settings']==policy['settings'] and c['resource_lock']==policy['resource_lock']
    assert c['mse_tolerance']==policy['mse_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert c['official_checkpoint_sequence_length']==512 and c['original_all_family_cases']==873
    assert c['original_all_label_views']==2037


@pytest.mark.parametrize('field',['cases','required_family_native_cases','required_family_label_views','settings','resource_lock','mse_tolerance','official_checkpoint_sequence_length'])
def test_smaller_or_relaxed_native_contract_cannot_be_registered(field):
    c=s.registration()
    if field=='cases':c[field].pop()
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=28*2**30
    elif field=='resource_lock':c[field]='different_lane'
    elif field=='mse_tolerance':c[field]['atol']=.001
    else:c[field]=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.mark.parametrize('field',['seed','window','epochs','batch','padding','device','threads','entrypoint'])
def test_original_zero_shot_model_and_external_context_must_remain(field):
    c,case,q,j,m=select()
    if field=='seed':case['seed']=43
    elif field=='window':case['window']=512
    elif field=='epochs':m['parameters']['epochs']=1
    elif field=='batch':m['parameters']['batch_size']=2
    elif field=='padding':m['parameters']['padding_policy']='minimal_patch'
    elif field=='device':j['parameter_overrides']['device']='cuda'
    elif field=='threads':q['evaluation']['cpu_threads']=1
    else:m['entrypoint']='unrelated_model'
    with pytest.raises(ValueError):s.validate_model(c,case,q,j,m)


def audit_fixture(model):
    p=model['parameters']
    return dict(purpose='full_frozen_pretrained_zero_shot_tsad',training_seconds=0.,training_losses=[],parameter_count=35341512,
        checkpoint_load_audit=dict(parameter_count=35341512,checkpoint_sha256=p['checkpoint_sha256'],config_sha256=p['config_sha256'],
            all_keys_strictly_loaded=True,configured_sequence_length=512,task='reconstruction',random_reinitialized_head=False,
            hub_downloads=False,fine_tuning_epochs=0,fit_points_used_for_parameter_estimation=0))


def test_TAB100_uses_official512_metadata_without_changing_external_window():
    c,case,q,j,m=select(100);r=audit_fixture(m)
    assert r['checkpoint_load_audit']['configured_sequence_length']!=j['window'] # The known old false predicate.
    assert s.registered_length(m)==512
    s.validate_model(c,case,q,j,m);s.validate_release_audit(r,m)
    assert j['window']==m['parameters']['window']==100 and m['parameters']['padding_policy']=='TAB_always_extra_patch'


@pytest.mark.parametrize('field',['training_seconds','training_losses','parameter_count','checkpoint_sha256','config_sha256','all_keys_strictly_loaded','configured_sequence_length','task','random_reinitialized_head','hub_downloads','fine_tuning_epochs','fit_points_used_for_parameter_estimation'])
def test_nonofficial_or_fitted_or_wrong_sequence_release_audit_is_rejected(field):
    _,_,_,_,m=select();r=audit_fixture(m);a=r['checkpoint_load_audit']
    if field=='training_seconds':r[field]=1.
    elif field=='training_losses':r[field]=[1.]
    elif field=='parameter_count':r[field]=1
    elif field in ['checkpoint_sha256','config_sha256']:a[field]='wrong'
    elif field=='all_keys_strictly_loaded':a[field]=False
    elif field=='configured_sequence_length':a[field]=100
    elif field=='task':a[field]='classification'
    elif field in ['random_reinitialized_head','hub_downloads']:a[field]=True
    else:a[field]=1
    with pytest.raises(ValueError):s.validate_release_audit(r,m)


def test_original_native_numerical_code_object_is_reused_without_edit():
    from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as old
    body=s.unchanged_native_body(lambda runtime,case:None,lambda *args:None)
    assert body.__code__ is old.native_worker.__code__
    assert body.__globals__ is not old.native_worker.__globals__
    assert body.__globals__['groups'] is old.groups


def test_streaming_parameter_digest_covers_all_cpu_weight_bytes_and_detects_changes():
    import torch,hashlib
    from types import SimpleNamespace
    model=SimpleNamespace(model=torch.nn.Linear(7,40000).requires_grad_(False))
    before=s.frozen_named_parameters(model)
    assert set(before)=={'weight','bias'}
    for name,p in model.model.named_parameters():
        assert before[name]['sha256']==hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
        assert before[name]['elements']==p.numel()
    with torch.no_grad():model.model.weight[0,0]+=1
    assert s.frozen_named_parameters(model)!=before
    model.model.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):s.frozen_named_parameters(model)


REAL_SMALL_SCRIPT=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_original_full18_moment_v1 as s
from scripts.flow_matching import run_moment_pretrained_job as source
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];s.ROOT=root;s.CONFIG='fixture_runtime.json';source.ROOT=root
meta=s.read(root/'fixture_metadata.json');model=meta['model'];evaluation=meta['evaluation'];dataset=meta['dataset'];case=meta['case'];job=meta['job'];config=meta['runtime']
if phase=='native':
 rng=np.random.default_rng(42);arrays={}
 for entity in dataset['entities']:
  p=entity['array_prefix']
  for role,n in [('train',201),('validation',105),('test',205)]:arrays[p+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
  y=np.zeros(205,np.int8);y[20:50]=1;arrays[p+'_labels']=y
 np.savez_compressed(root/'fixture_inputs.npz',**arrays);dataset['input_sha256']=s.sha(root/'fixture_inputs.npz')
 s.write(root/'fixture_model.json',model)
 for n,policy in enumerate(dataset['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in dataset['entities']}
  if n:labels['a_labels'][50+n]=1
  target=root/(policy+'.npz');np.savez_compressed(target,**labels)
  dataset['labels_only_controls'][policy].update(path=str(target),sha256=s.sha(target))
 s.write(root/'fixture_manifest.json',dict(datasets=[dataset]));s.write(root/'fixture_runtime.json',config)
 effective=dict(evaluation,bootstrap_block_points=max(evaluation['bootstrap_block_points'],job['window']))
 source.run(job,effective)
 result=s.read(root/'source/result.json');binding=dict(fixture_only=True,result_sha256=s.sha(root/'source/result.json'),source_guard_sha256='1'*64)
 with np.load(root/'fixture_inputs.npz',allow_pickle=False) as inputs,np.load(root/'source/scores.npz',allow_pickle=False) as saved:
  entities,val,train,coverage,digests=s.scalar.collect_arrays(inputs,saved,dataset)
 overlays={}
 for name,label in dataset['labels_only_controls'].items():
  with np.load(label['path'],allow_pickle=False) as saved:overlays[name]={k:saved[k] for k in saved.files}
 views=s.scalar.evaluate_views(entities,val,train,overlays,dataset,result,effective,42)
 s.write(root/'fixture_scalar.json',dict(label_view_results=views,source_array_digests=digests,full_positional_saved_score_coverage=coverage))
 meta.update(dataset=dataset,binding=binding);(root/'fixture_metadata.json').write_text(json.dumps(meta))
else:result=s.read(root/'source/result.json');binding=meta['binding']
# Private unit-process providers isolate these tiny-input mathematical bodies
# from production job gates;no synthetic source guard or benchmark is created.
proof=dict(path=str(root/'fixture_scalar.json'),sha256=s.sha(root/'fixture_scalar.json'),fixture_only=True)
s.scalar.source_binding=lambda config,case:(dict(evaluation=evaluation),job,result,dataset,binding)
s.ready_scalar=lambda config,case:proof
if phase=='native':s.native_worker(config,case)
else:
 _,native_dir=s.paths(config,case,'native');native_path=native_dir/s.REPORTS['native']
 s.ready_native=lambda config,case:(native_path,s.read(native_path))
 s.review_worker(config,case)
_,output=s.paths(config,case,phase);r=s.read(output/s.REPORTS[phase]);assert r['passed']
if phase=='native':
 assert r['parameters']==35341512 and sum(v['elements'] for v in r['frozen_named_parameter_digests'].values())==35341512
 assert r['all_frozen_named_parameters_identical_before_and_after'] and r['original_native_worker_code_unchanged']
 assert r['all_array_dtype_shape_content_bytes_identical'] and r['official_checkpoint_loader_fit_calls']==1
 assert r['parameter_estimation_fit_calls']==r['additional_optimizer_updates']==0 and not r['CUDA_context_initialized']
 assert r['official_checkpoint_length']==512 and r['external_context']==job['window']
 expected=16 if job['window']==100 else 6
 assert r['inline_full_window_MSE_check']['forward_calls']==expected
 assert r['inline_full_window_MSE_check']['windows']==expected
else:assert r['all_original_labels_metrics_events_bootstrap_and_grid_recomputed'] and len(r['all_label_view_results'])==4
'''


@pytest.mark.parametrize('window',[100,512])
def test_real_official_Small_weights_both_contexts_native_and_separate_process_review(tmp_path,window):
    c,actual,q,old,model=select(window)
    case=dict(id='official_small_tiny_input_fixture_not_benchmark',family='moment',seed=42,window=window,
        worker_python=actual['worker_python'],source_guard='unit_provider_only_not_real_guard',queue_path='unit_fixture_queue_not_production.json')
    job=dict(id=case['id'],dataset='fixture',input_kind='standard',seed=42,window=window,model_config='fixture_model.json',dataset_manifest_path='fixture_manifest.json',
        parameter_overrides=dict(device='cpu',seed=42),output_directory='source',frozen_files=[])
    data=dict(id='fixture',input_npz='fixture_inputs.npz',grouped_splits_verified=True,primary_label_policy='source_first_exclusive',
        entities=[dict(entity_id='a',array_prefix='a'),dict(entity_id='b',array_prefix='b')],
        labels_only_controls={p:{} for p in ['source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive']})
    runtime=dict(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),expected_source_model_entrypoint=c['expected_source_model_entrypoint'],mse_tolerance=c['mse_tolerance'])
    metadata=dict(model=model,case=case,job=job,dataset=data,evaluation=q['evaluation'],runtime=runtime,fixture_only_not_benchmark_performance=True)
    (tmp_path/'fixture_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for phase in ['native','review']:
        p=subprocess.run([str(s.ROOT/actual['worker_python']),'-B','-X','utf8','-c',REAL_SMALL_SCRIPT,str(tmp_path),phase],
            cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=180)
        assert p.returncode==0,p.stdout[-1500:]+p.stderr[-5000:]


def test_cold_registration_identical_in_author310_and_controller312():
    c=s.registration();python=c['cases'][0]['worker_python']
    code='from scripts.flow_matching import replay_original_full18_moment_v1 as s;import json,sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules;print(json.dumps(c,sort_keys=True))'
    p=subprocess.run([str(s.ROOT/python),'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    assert json.loads(p.stdout)==c
