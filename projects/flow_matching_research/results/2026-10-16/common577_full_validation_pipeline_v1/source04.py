"""New binding tests;synthetic CPU checkpoints never certify CUDA/benchmark."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import replay_common105_matched577_v1 as s


def test_all577_models1726labels_and_original_fidelity_resource_contracts():
    total=views=0
    for family in s.SPECS:
        c=s.family_registration(family);s.verify_family(family,c);module=s.component(family);old=s.read(s.ROOT/module.CONFIG)
        total+=len(c['cases']);views+=c['required_family_label_views']
        for key in ['settings','resource_lock','math_tolerance','mse_tolerance','GPU_settings','GPU_resource_lock',
            'gpu_prerequisite_queues','legacy_gpu_queue_lock','expected_source_device','source_cpu_threads',
            'source_model_configurations','expected_source_model_entrypoint','expected_source_model_entrypoints']:
            if key in old:assert c[key]==old[key]
        assert c['source_execution']==s.scalar.source.CONFIG
        assert c['upstream_scalar_configuration']==s.scalar.CONFIG
        for case in c['cases']:
            q,j=s.scalar.original.resolve_case(case)
            if family in ['fm','neural','baselines','moment']:
                module.validate_model(c,case,q,j,s.read(s.ROOT/j['model_config']))
    assert (total,views)==(577,1726)


@pytest.mark.parametrize('family',list(s.SPECS))
def test_native_review_and_readiness_code_objects_are_the_complete_originals(family):
    module=s.component(family);ns=s.namespace(family)
    for name in ['native_worker','review_worker','ready_native']:
        assert ns[name].__code__ is getattr(module,name).__code__
        assert ns[name].__globals__ is ns
    assert ns['CONFIG']==s.CONFIGS[family] and ns['scalar'].source_binding is s.scalar.source_binding
    assert ns['ready_scalar'] is s.ready_scalar and ns['paths'] is s.paths
    c=s.family_registration(family);fn=s.bound_controller(family)
    assert c['required_family_native_cases'] in fn.__code__.co_consts
    assert c['required_family_label_views'] in fn.__code__.co_consts
    assert fn.__globals__['worker_command'](c['cases'][0],'native')==s.worker_command(family,c['cases'][0],'native')


@pytest.mark.parametrize('fault',['cases','labels','scope','memory','lock','math','model','device','threads','GPU_gate'])
def test_narrowed_family_changed_model_or_weakened_policy_rejected(fault):
    family='neural';c=s.family_registration(family)
    if fault=='cases':c['cases'].pop()
    elif fault=='labels':c['required_family_label_views']=1
    elif fault=='scope':c['source_execution']=s.scalar.original.source.CONFIG
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='lock':c['resource_lock']='other.lock'
    elif fault=='math':c['math_tolerance']['atol']=.01
    elif fault=='model':c['source_model_configurations'].clear()
    elif fault=='device':c['expected_source_device']='cpu'
    elif fault=='threads':c['source_cpu_threads']=1
    else:c['gpu_prerequisite_queues']=[]
    with pytest.raises(ValueError):s.verify_family(family,c)


def test_every_worker_command_preserves_actual_interpreter_and_explicit_native_family():
    for family in s.SPECS:
        c=s.family_registration(family)
        for case in c['cases']:
            command=s.worker_command(family,case,'native')
            assert command[0]==str(s.ROOT/case['worker_python'])
            assert command[command.index('--family')+1]==family and command[-2:]==['--stage','native']
    with pytest.raises(ValueError):s.worker_command('fm',c['cases'][0],'fit')


FIXTURE=r'''
import json,pathlib,sys,numpy as np,torch
from scripts.flow_matching import replay_common105_matched577_v1 as s
from scripts.flow_matching import run_tsad_experiment as producer
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];m=s.read(root/'fixture_metadata.json');family=m['family']
s.ROOT=root;s.CONFIGS[family]='fixture_native.json';c=m['runtime'];case=m['case'];job=m['job'];data=m['dataset'];evaluation=m['evaluation']
if phase=='native':
 rng=np.random.default_rng(1103);arrays={};window=case['window']
 for entity in data['entities']:
  p=entity['array_prefix']
  for role,n in [('train',window+2),('validation',window),('test',window*2+7)]:arrays[p+'_'+role]=rng.normal(size=(n,2)).astype(np.float32)
  y=np.zeros(window*2+7,np.uint8);y[20:40]=1;arrays[p+'_labels']=y
 np.savez_compressed(root/'inputs.npz',**arrays);data['input_sha256']=s.sha(root/'inputs.npz')
 s.write(root/'manifest.json',dict(datasets=[data]));s.write(root/'model.json',m['model']);s.write(root/'fixture_native.json',c)
 torch.set_num_threads(evaluation['cpu_threads']);producer.run(root,job,evaluation)
 result=s.read(root/'source/result.json');binding=dict(fixture_only=True,result_sha256=s.sha(root/'source/result.json'))
 for index,policy in enumerate(data['labels_only_controls']):
  labels={e['array_prefix']+'_labels':arrays[e['array_prefix']+'_labels'].copy() for e in data['entities']}
  if index:labels['a_labels'][40:40+index]=1
  path=root/(policy+'.npz');np.savez_compressed(path,**labels)
  data['labels_only_controls'][policy].update(path=str(path),sha256=s.sha(path))
 with np.load(root/'inputs.npz',allow_pickle=False) as inputs,np.load(root/'source/scores.npz',allow_pickle=False) as saved:
  entities,val,train,coverage,digests=s.scalar.original.collect_arrays(inputs,saved,data)
 overlays={}
 for policy,label in data['labels_only_controls'].items():
  with np.load(label['path'],allow_pickle=False) as labels:overlays[policy]={k:labels[k] for k in labels.files}
 views=s.scalar.original.evaluate_views(entities,val,train,overlays,data,result,evaluation,1103)
 s.write(root/'fixture_scalar.json',dict(source_array_digests=digests,full_positional_saved_score_coverage=coverage,label_view_results=views))
 m.update(dataset=data,binding=binding);(root/'fixture_metadata.json').write_text(json.dumps(m))
else:result=s.read(root/'source/result.json');binding=m['binding']
# These providers are isolated to this fixture child;no scientific guards.
s.scalar.source_binding=lambda config,case:(dict(evaluation=evaluation),job,result,data,binding)
s.ready_scalar=lambda config,case:dict(path=str(root/'fixture_scalar.json'),sha256=s.sha(root/'fixture_scalar.json'),fixture_only=True)
ns=s.namespace(family)
if phase=='review':
 _,native_dir=s.paths(c,case,'native');file=native_dir/ns['REPORTS']['native']
 ns['ready_native']=lambda config,case:(file,s.read(file))
ns['native_worker' if phase=='native' else 'review_worker'](c,case)
_,output=s.paths(c,case,phase);proof=s.read(output/ns['REPORTS'][phase]);assert proof['passed']
if phase=='native':
 assert proof['all_array_dtype_shape_content_bytes_identical'] and proof['all_model_tensors_identical_before_and_after']
 assert proof['additional_fit_calls']==proof['additional_optimizer_updates']==0
 assert len(proof['full_native_coverage'])==6 and not proof['CUDA_context_initialized']
else:
 assert proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
 assert len(proof['all_label_view_results'])==len(case['required_label_policies'])
'''


@pytest.mark.parametrize('family',['baselines','fm'])
def test_actual_original_local_budget_checkpoint_reloads_under_new_binding_in_fresh_processes(tmp_path,family):
    c=s.family_registration(family)
    for original in c['cases']:
        q,j=s.scalar.original.resolve_case(original);model=s.read(s.ROOT/j['model_config'])
        if family=='baselines' and model['parameters']['method']=='dcdetector':break
        if family=='fm' and model['parameters']['objective']=='independent':break
    else:raise AssertionError('Original required representative absent')
    case=dict(id='fixture_only_not_benchmark',family=original['family'],dataset='fixture',seed=1103,window=j['window'],worker_python=sys.executable,
        required_label_policies=original['required_label_policies'])
    job=dict(id=case['id'],dataset='fixture',seed=1103,window=case['window'],model_config='model.json',dataset_manifest_path='manifest.json',
        parameter_overrides=dict(seed=1103,device='cpu'),output_directory='source',frozen_files=[])
    data=dict(id='fixture',input_npz='inputs.npz',entities=[dict(entity_id=x,array_prefix=x,features=2) for x in ['a','b']],
        labels_only_controls={p:{} for p in case['required_label_policies']},primary_label_policy=case['required_label_policies'][0],grouped_splits_verified=True)
    runtime=deepcopy(c);runtime.update(artifact_root=str(tmp_path/'replay'),state_root=str(tmp_path/'state'),cases=[case],source_model_configurations={'model.json':model})
    s.write(tmp_path/'fixture_metadata.json',dict(family=family,model=model,case=case,job=job,dataset=data,runtime=runtime,evaluation=q['evaluation'],
        synthetic_only_not_benchmark_or_CUDA=True,original_local_capacity_and_epoch_budget_unchanged=True))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(s.ROOT)+os.pathsep+str(s.ROOT/'src'),
        OMP_NUM_THREADS=str(q['evaluation']['cpu_threads']),MKL_NUM_THREADS=str(q['evaluation']['cpu_threads']))
    for phase in ['native','review']:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-c',FIXTURE,str(tmp_path),phase],cwd=s.ROOT,env=env,capture_output=True,text=True,timeout=240)
        assert p.returncode==0,p.stdout[-1200:]+p.stderr[-4000:]


def test_both_producer_python_versions_verify_every_family_cold():
    source=s.read(s.ROOT/s.scalar.CONFIG)
    code='from scripts.flow_matching import replay_common105_matched577_v1 as s;import sys;[(s.verify_family(f,s.family_registration(f))) for f in s.SPECS];assert "numpy" not in sys.modules and "torch" not in sys.modules'
    for exe in sorted({c['worker_python'] for c in source['cases']}):
        p=subprocess.run([str(s.ROOT/exe),'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
        assert p.returncode==0,p.stdout+p.stderr
