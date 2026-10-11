"""All-eight replay contracts and independent restoration fixtures; no benchmark performance."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import run_gtgan_all8_full_native_replay_v1 as replay


SOURCE=replay.ROOT/'data/public_datasets/flow_matching/gtgan_native_source_v1/original'
QUEUES={'stock':'configs/experiments/fm_gtgan_native_original_stock_queue.v1.json',
    'energy':'configs/experiments/fm_gtgan_native_original_energy_queue.v1.json'}


def full_config():
    routes=[]
    for dataset,path in QUEUES.items():
        q=replay.read(replay.ROOT/path)
        for j in q['jobs']:
            r=dict(dataset=dataset,original_queue=path,canonical_job=j,settings=q['settings'],
                result_path=(Path(j['output_directory'])/'result.json').as_posix(),native_completion_receipt='fixture-native-receipt.json')
            b=replay.budget(r)
            r.update(full_generation_shape=[b['full_generation_windows'],24,b['variables']],
                full_noise_shape=[b['full_generation_windows'],24,b['latent_noise_variables']],
                full_periodic_evaluations=b['full_periodic_evaluations'],full_trained_evaluators=20*b['full_periodic_evaluations'],
                original_truth_audit=dict(full_shape=[b['full_generation_windows'],24,b['variables']]))
            routes.append(r)
    q=replay.read(replay.ROOT/QUEUES['stock'])
    return dict(routes=routes,original_queues=QUEUES,source_receipts=[],settings=q['settings'],python=q['python'],resource_lock=q['resource_lock'],
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        equality_gate='bitwise',additional_generator_training_seeds=0,required_full_generations=60,
        required_full_trained_evaluators=1200,required_native_network_checkpoint_artifacts=236,
        required_full_generated_windows=717270,cpu_threads=2)


def test_all_eight_actual_original_commands_full_shapes_and_complete_obligations():
    c=full_config();assert replay.verify(c)==c['routes']
    assert len(c['routes'])==8
    budgets=[replay.budget(r) for r in c['routes']]
    assert sum(v['full_periodic_evaluations'] for v in budgets)==60
    assert sum(v['full_periodic_evaluations']*20 for v in budgets)==1200
    assert sum(v['full_periodic_evaluations']*v['full_generation_windows'] for v in budgets)==717270
    assert sum(2 if v['native_release_checkpoint_evaluation'] else 4*v['full_periodic_evaluations'] for v in budgets)==236
    assert sum(v['native_release_checkpoint_evaluation'] for v in budgets)==2
    assert all(v['latent_noise_variables']==24 for v in budgets)
    for r,b in zip(c['routes'],budgets):
        assert b['variables']==(6 if r['dataset']=='stock' else 28)
        assert b['discriminative_evaluator_steps']==(2000 if r['dataset']=='stock' else 300)
        assert b['predictive_evaluator_steps']==5000
        assert b['full_generation_windows']==(3662 if r['dataset']=='stock' else 19712)
        assert r['canonical_job']['id'].endswith('seed7777')


@pytest.mark.parametrize('key,value',[('equality_gate','allclose'),('additional_generator_training_seeds',1),
    ('required_full_generations',1),('required_full_trained_evaluators',20),('required_native_network_checkpoint_artifacts',4),
    ('required_full_generated_windows',3662),('cpu_threads',1),('original_scope',{'papers':1})])
def test_reduced_or_relaxed_replay_obligations_are_rejected(key,value):
    c=full_config();c[key]=value
    with pytest.raises(ValueError):replay.verify(c)


@pytest.mark.parametrize('field,value',[('full_noise_shape',[3662,24,6]),('full_generation_shape',[128,24,6]),
    ('full_periodic_evaluations',1),('full_trained_evaluators',20),('native_completion_receipt','')])
def test_full_route_input_shape_cadence_and_real_completion_receipt_required(field,value):
    c=full_config();c['routes'][0][field]=value
    with pytest.raises(ValueError):replay.verify(c)


@pytest.mark.parametrize('kind',['missing','duplicate','different_job','separate_lock','reduced_guard'])
def test_no_original_command_or_shared_resource_gate_can_be_dropped(kind):
    c=full_config()
    if kind=='missing':c['routes'].pop()
    elif kind=='duplicate':c['routes'][0]=copy.deepcopy(c['routes'][1])
    elif kind=='different_job':c['routes'][0]['canonical_job']=dict(c['routes'][0]['canonical_job'],case_id='different')
    elif kind=='separate_lock':c['resource_lock']='different.lock'
    else:c['settings']=dict(c['settings'],minimum_commit_headroom_bytes=1)
    with pytest.raises((ValueError,StopIteration)):replay.verify(c)


@pytest.mark.parametrize('role,name,filename,outputs',[
    ('discriminative','discriminative_score_metrics','discriminative_metrics.py',['discriminator/Sigmoid:0','discriminator_1/Sigmoid:0']),
    ('predictive','predictive_score_metrics','predictive_metrics.py',['predictor/Sigmoid:0']),
    ('predictive','predictive_score_metrics2','predictive_metrics.py',['predictor/Sigmoid:0'])])
def test_original_metric_scope_and_native_prediction_tensor_names(role,name,filename,outputs):
    source=(SOURCE/'metrics'/filename).read_text(encoding='utf-8')
    assert replay.metric_output_contract(role,name,source)==outputs
    changed=source.replace('tf.nn.sigmoid(y_hat_logit)','tf.nn.sigmoid(y_hat_logit,name="changed")')
    with pytest.raises(ValueError):replay.metric_output_contract(role,name,changed)


def test_every_original_energy_generator_and_recovery_mode_pair_is_proven():
    source=(SOURCE/'GTGAN_energy.py').read_text(encoding='utf-8')
    assert replay.energy_mode_contract(source)['original_mode_pairs']==4
    with pytest.raises(ValueError):replay.energy_mode_contract(source.replace('recovery.eval()','recovery.train()',1))


def test_stock_and_energy_keep_their_distinct_actual_original_sampler_functions():
    for filename,name in [('GTGAN_stocks.py','run_latent_ctfp_model5_3'),('GTGAN_stocks_irregular.py','run_latent_ctfp_model5_3'),
            ('GTGAN_energy.py','run_latent_ctfp_model5_2')]:
        tree=ast.parse((SOURCE/filename).read_text(encoding='utf-8'))
        imports=[alias.name for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='ctfp_tools'
            for alias in n.names if alias.asname=='run_model']
        assert imports==[name]
    sampler=next(n for n in ast.parse((SOURCE/'ctfp_tools.py').read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name=='run_latent_ctfp_model5_3')
    first=next(n for n in sampler.body if isinstance(n,ast.If))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='sample_standard_gaussian'
        for statement in first.body for n in ast.walk(statement))


def test_entire_original_energy_verifier_is_identical_without_host_only_dependencies():
    import inspect
    path=replay.ROOT/'scripts/flow_matching/run_gtgan_energy_native_queue_v1.py'
    source=path.read_text(encoding='utf-8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='verify_result')
    function=replay.compiled_energy_verifier()
    assert ast.dump(ast.parse(inspect.getsource(function)).body[0])==ast.dump(node)
    assert 'psutil' not in function.__globals__ and 'full_gpu_memory_slot' not in function.__globals__


def test_complete_original_energy_budget_function_is_preserved_without_indirect_scheduler_imports():
    import inspect
    path=replay.ROOT/'scripts/flow_matching/gtgan_energy_native_common_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='full_budget')
    assert ast.dump(ast.parse(inspect.getsource(replay.energy_budget)).body[0])==ast.dump(node)
    assert 'psutil' not in replay.energy_budget.__globals__


def test_all_native_truth_window_bytes_and_duplicate_counts_survive_shuffling():
    source=np.arange(5*24*6,dtype=np.float32).reshape(5,24,6);source[4]=source[1]
    replay.exact_window_multiset(source[[4,0,3,1,2]],source,np)
    altered=source.copy();altered[4]=source[2]
    with pytest.raises(ValueError):replay.exact_window_multiset(altered,source,np)
    with pytest.raises(ValueError):replay.exact_window_multiset(source[:-1],source,np)
    with pytest.raises(ValueError):replay.exact_window_multiset(source.astype(np.float64),source,np)


def test_native_truth_audit_uses_the_original_float32_loader_cast_without_relaxed_gate(tmp_path):
    truth=np.arange(5*24*6,dtype=np.float64).reshape(5,24,6)/1000
    audit=tmp_path/'audit.npz';np.savez_compressed(audit,truth=truth)
    generation=tmp_path/'generated.npz';np.savez_compressed(generation,real=truth.astype(np.float32)[[4,2,0,1,3]])
    route={'original_truth_audit':dict(artifact_path=str(audit),artifact_sha256=replay.sha(audit),full_shape=[5,24,6])}
    assert replay.verify_native_truth({'artifact_path':str(generation)},route,np)['no_truth_window_omitted']
    np.savez_compressed(generation,real=truth.astype(np.float32)[:4])
    with pytest.raises(ValueError):replay.verify_native_truth({'artifact_path':str(generation)},route,np)


def test_bitwise_comparison_rejects_a_single_native_float_bit_change():
    source=np.ones((5,24,6),dtype=np.float32);actual=source.copy()
    replay.exact_arrays(actual,source,'fixture',np)
    actual[0,0,0]=np.nextafter(actual[0,0,0],np.float32(2))
    with pytest.raises(ValueError):replay.exact_arrays(actual,source,'fixture',np)


def test_bitwise_comparison_rejects_signed_zero_while_accepting_identical_noncontiguous_values():
    source=np.zeros((5,24,6),dtype=np.float32);actual=source.copy()
    actual[0,0,0]=np.float32(-0.0)
    assert np.array_equal(actual,source)
    with pytest.raises(ValueError):replay.exact_arrays(actual,source,'signed zero',np)
    replay.exact_arrays(source.transpose(0,2,1),source.transpose(0,2,1).copy(),'all bytes',np)


def test_full_state_digest_includes_parameter_names_shapes_dtypes_and_all_values():
    before={'weight':np.arange(5,dtype=np.float32)}
    digest=replay.tensor_digest(before,np)
    assert digest==replay.tensor_digest(copy.deepcopy(before),np)
    for changed in [{'other':before['weight']},{'weight':before['weight'].astype(np.float64)},
        {'weight':before['weight'].reshape(1,5)},{'weight':before['weight']+1}]:
        assert digest!=replay.tensor_digest(changed,np)


def operation(name,kind='Placeholder',inputs=(),controls=()):
    return SimpleNamespace(name=name,type=kind,inputs=list(inputs),control_inputs=list(controls))


def test_native_prediction_dependency_trace_rejects_optimizer_and_control_updates():
    variable=operation('variable','VariableV2');input=SimpleNamespace(op=variable)
    pred=operation('prediction','Sigmoid',[input])
    assert replay.prediction_reachable_ops([SimpleNamespace(op=pred)])==2
    update=operation('optimizer','ApplyAdam')
    pred.control_inputs.append(update)
    with pytest.raises(ValueError):replay.prediction_reachable_ops([SimpleNamespace(op=pred)])


def evaluator_fixture(tmp_path,role,predictive_outputs=1,energy=False):
    n=7;variables=28 if energy else 6
    expected=dict(full_generation_windows=n,variables=variables,
        predictive_output_variables=predictive_outputs,predictive_function='predictive_score_metrics2' if predictive_outputs==1 else 'predictive_score_metrics',
        discriminative_evaluator_steps=300 if energy else 2000,predictive_evaluator_steps=5000)
    source=(SOURCE/'metrics'/('discriminative_metrics.py' if role=='discriminative' else 'predictive_metrics.py')).read_text(encoding='utf-8')
    if role=='discriminative':
        m=n-int(.8*n);predictions=[np.ones((m,1),dtype=np.float32)*.8,np.ones((m,1),dtype=np.float32)*.2]
        arrays={'myinput_x:0':np.zeros((m,24,variables),dtype=np.float32),'myinput_x_hat:0':np.zeros((m,24,variables),dtype=np.float32),
            'myinput_t:0':np.ones(m,dtype=np.int64)*24,'myinput_t_hat:0':np.ones(m,dtype=np.int64)*24,
            'prediction_real':predictions[0],'prediction_fake':predictions[1]};score=.5;function='discriminative_score_metrics'
    else:
        predictions=[np.ones((n,23,predictive_outputs),dtype=np.float32)*.125]
        arrays={'myinput_x:0':np.zeros((n,23,variables if predictive_outputs==variables else variables-1),dtype=np.float32),
            'myinput_t:0':np.ones(n,dtype=np.int64)*23,'prediction':predictions[0],
            'original_reference':np.zeros((n,23,predictive_outputs),dtype=np.float32)};score=.125;function=expected['predictive_function']
    array_path=tmp_path/'original_test_arrays.npz';np.savez_compressed(array_path,**arrays)
    checkpoints=[]
    for suffix in ['.meta','.index','.data-00000-of-00001']:
        path=tmp_path/('native_evaluator'+suffix);path.write_bytes(b'fixture-only-never-native-weights')
        checkpoints.append(dict(path=str(path),sha256=replay.sha(path)))
    evaluator=dict(role=role,evaluator_function=function,optimizer_updates=expected[role+'_evaluator_steps'],
        expected_optimizer_updates=expected[role+'_evaluator_steps'],evaluator_training_instances=1,
        additional_generator_training_seeds=0,checkpoint_files=checkpoints,array_path=str(array_path),
        array_sha256=replay.sha(array_path),trained_variable_count=2,full_saved_predictions_recomputed_score=score,original_returned_score=score)
    names=replay.metric_output_contract(role,function,source)
    calls=[];variable_values=[np.array([.25],dtype=np.float32),np.array([5000],dtype=np.int64)]
    variables_list=[SimpleNamespace(name='original_weight:0'),SimpleNamespace(name='original_optimizer:0')]
    class Key:
        def __init__(self,name):self.name=name;self.op=operation(name)
    lookup={key:Key(key) for key in arrays if key.endswith(':0')}
    for name in names:lookup[name]=Key(name)
    class Graph:
        def as_default(self):return nullcontext()
        def get_tensor_by_name(self,name):return lookup[name]
    class Session:
        def __init__(self,graph,config):assert config=={'intra_op_parallelism_threads':2,'inter_op_parallelism_threads':1,'device_count':{'GPU':0}}
        def __enter__(self):return self
        def __exit__(self,*args):calls.append('close')
        def run(self,fetches,feed_dict=None):
            if fetches is variables_list:calls.append('all_model_optimizer_variables');return variable_values
            assert [v.name for v in fetches]==names
            assert {key.name for key in feed_dict}=={key for key in arrays if key.endswith(':0')}
            for key,value in feed_dict.items():assert np.array_equal(value,arrays[key.name])
            calls.append('full_native_predictions');return [np.array(v,copy=True) for v in predictions]
    class Saver:
        def restore(self,session,prefix):assert prefix.endswith('native_evaluator');calls.append('restore_entire_native_checkpoint')
    tf=SimpleNamespace(Graph=Graph,compat=SimpleNamespace(v1=SimpleNamespace(
        Session=Session,ConfigProto=lambda **kw:kw,global_variables=lambda:variables_list,
        train=SimpleNamespace(import_meta_graph=lambda path,clear_devices:Saver()))))
    return evaluator,expected,source,tf,calls,lookup


@pytest.mark.parametrize('role,out,energy',[
    ('discriminative',1,False),('predictive',1,False),('predictive',6,False),
    ('discriminative',1,True),('predictive',1,True),('predictive',28,True)])
def test_independent_full_trained_graph_restore_feeds_predictions_and_optimizer_state(tmp_path,role,out,energy):
    e,b,s,tf,calls,_=evaluator_fixture(tmp_path,role,out,energy)
    report=replay.replay_evaluator(e,b,s,tmp_path/'replay',np,tf)
    assert calls==['restore_entire_native_checkpoint','all_model_optimizer_variables','full_native_predictions','all_model_optimizer_variables','close']
    assert report['full_predictions_bitwise_equal'] and report['native_model_and_optimizer_state_unchanged']
    assert report['numerical_optimizer_updates']==report['additional_generator_training_seeds']==0
    assert report['all_original_model_and_optimizer_variables_restored']==2
    assert Path(report['replay_artifact_path']).is_file()


@pytest.mark.parametrize('bad',['prediction_bit','model_optimizer_state'])
def test_restored_model_prediction_or_optimizer_state_changes_fail_full_replay(tmp_path,bad):
    e,b,s,tf,calls,_=evaluator_fixture(tmp_path,'predictive')
    original=tf.compat.v1.Session
    class ChangedSession(original):
        def __init__(self,*a,**k):super().__init__(*a,**k);self.variable_reads=0
        def run(self,fetches,feed_dict=None):
            values=super().run(fetches,feed_dict)
            if feed_dict is not None:
                if bad=='prediction_bit':values[0][0,0,0]=np.nextafter(values[0][0,0,0],np.float32(1))
            else:
                self.variable_reads+=1
                if bad=='model_optimizer_state' and self.variable_reads==2:
                    values=[np.array(v,copy=True) for v in values];values[1]+=1
            return values
    tf.compat.v1.Session=ChangedSession
    with pytest.raises(ValueError):replay.replay_evaluator(e,b,s,tmp_path/'replay',np,tf)


@pytest.mark.parametrize('bad',['step_count','additional_seed','component_missing','checkpoint_tampered','feed_tampered',
    'partial_feed','partial_prediction','optimizer_dependency','function_branch','wrong_variable_count'])
def test_incomplete_or_changed_native_trained_evaluator_inputs_are_rejected(tmp_path,bad):
    e,b,s,tf,calls,lookup=evaluator_fixture(tmp_path,'predictive')
    if bad=='step_count':e['optimizer_updates']=4999
    elif bad=='additional_seed':e['additional_generator_training_seeds']=1
    elif bad=='component_missing':e['checkpoint_files'].pop()
    elif bad=='checkpoint_tampered':Path(e['checkpoint_files'][0]['path']).write_bytes(b'changed')
    elif bad=='feed_tampered':Path(e['array_path']).write_bytes(b'changed')
    elif bad in ['partial_feed','partial_prediction']:
        with np.load(e['array_path']) as z:arrays={key:z[key] for key in z.files}
        key='myinput_x:0' if bad=='partial_feed' else 'prediction';arrays[key]=arrays[key][:2]
        np.savez_compressed(e['array_path'],**arrays);e['array_sha256']=replay.sha(e['array_path'])
    elif bad=='optimizer_dependency':lookup['predictor/Sigmoid:0'].op.control_inputs=[operation('Adam','ResourceApplyAdam')]
    elif bad=='function_branch':e['evaluator_function']='predictive_score_metrics'
    else:e['trained_variable_count']=1
    with pytest.raises(ValueError):replay.replay_evaluator(e,b,s,tmp_path/'replay',np,tf)
    assert 'full_native_predictions' not in calls


def test_successful_native_completion_requires_zero_exit_no_guard_abort_and_verified_result(tmp_path):
    path=tmp_path/'receipt.json';route={'native_completion_receipt':str(path)}
    assert replay.successful_native_receipt(route) is None
    for data in [dict(exit_code=0,full_native_result_verified=True),dict(exit_code=1,resource_guard_aborted=False,full_native_result_verified=True),
        dict(exit_code=0,resource_guard_aborted=True,full_native_result_verified=True),dict(exit_code=0,resource_guard_aborted=False,full_native_result_verified=False)]:
        path.write_text(__import__('json').dumps(data))
        assert replay.successful_native_receipt(route) is None
    data=dict(exit_code=0,resource_guard_aborted=False,full_native_result_verified=True)
    path.write_text(__import__('json').dumps(data))
    assert replay.successful_native_receipt(route)==data


def test_each_full_training_period_and_original_release_network_checkpoints_are_distinct(tmp_path):
    expected={'native_release_checkpoint_evaluation':False};result={'checkpoints':[]}
    for step in [500,1000]:
        for role in ['embedder','generator','recovery','discriminator']:
            path=tmp_path/(role+str(step)+'.pt');path.write_bytes((role+str(step)).encode())
            result['checkpoints'].append(dict(path=str(path),sha256=replay.sha(path)))
    selected=replay.native_checkpoints({'native_joint_step':1000},result,expected)
    assert set(selected)=={'embedder','generator','recovery','discriminator'}
    assert all(Path(v['path']).stem.endswith('1000') for v in selected.values())
    result['released_checkpoints']=[]
    for role in ['generator','recovery']:
        path=tmp_path/(role+'.pt');path.write_bytes(b'original-release-fixture')
        result['released_checkpoints'].append(dict(path=str(path),sha256=replay.sha(path)))
    assert set(replay.native_checkpoints({'native_joint_step':0},result,{'native_release_checkpoint_evaluation':True}))=={'generator','recovery'}


class Tensor:
    def __init__(self,values):self.values=np.asarray(values)
    def detach(self):return self
    def cpu(self):return self
    def cuda(self):return self
    def numpy(self):return self.values
    def all(self):return Tensor(self.values.all())
    def item(self):return self.values.item()


def generation_fixture(tmp_path,energy=False,release=False):
    n=7;variables=28 if energy else 6;calls=[];states={}
    expected=dict(full_generation_windows=n,latent_noise_variables=24,native_release_checkpoint_evaluation=release,variables=variables)
    generated=np.arange(n*24*variables,dtype=np.float32).reshape(n,24,variables)/100
    model_roles=['embedder','generator','recovery','discriminator'];result={'checkpoints':[],'released_checkpoints':[]}
    for index,role in enumerate(model_roles):
        path=tmp_path/(role+('' if release else '500')+'.pt');path.write_bytes(('fixture '+role).encode())
        states[str(path)]={'weight':Tensor(np.array([index+.25],dtype=np.float32))}
        result['released_checkpoints' if release and role in ['generator','recovery'] else 'checkpoints'].append(
            dict(path=str(path),sha256=replay.sha(path)))
    class Model:
        def __init__(self,name):self.name=name;self.weight=Tensor(np.zeros(1,dtype=np.float32));self.training=True
        def load_state_dict(self,state,strict):assert strict;self.weight=Tensor(np.array(state['weight'].values,copy=True));calls.append('load_'+self.name)
        def state_dict(self):return {'weight':self.weight}
        def named_parameters(self):return [('weight',self.weight)]
        def train(self,mode):self.training=mode
        def __call__(self,h_hat,observations):
            assert self.name=='recovery';calls.append('original_full_recovery');return Tensor(generated)
    models={name:Model(name) for name in ['embedder','generator','supervisor','recovery','discriminator']}
    namespace=dict(models,args=SimpleNamespace(activation='exp',effective_shape=24),device='native-cuda-fixture')
    def native_sampler(args,model,noise,times,device,z):
        assert args is namespace['args'] and model is models['generator'] and device=='native-cuda-fixture' and z is True
        assert noise.values.shape==(n,24,24) and times.values.shape==(n,24,1)
        calls.append('original_full_sampler');return Tensor(np.zeros((n,24,24),dtype=np.float32))
    namespace['run_model']=native_sampler
    noise=np.arange(n*24*24,dtype=np.float32).reshape(n,24,24)
    times=np.broadcast_to(np.arange(24,dtype=np.float32)[None,:,None],(n,24,1)).copy()
    observations=np.broadcast_to(np.arange(24,dtype=np.float32)[None,:],(n,24)).copy()
    cpu=np.array([1,2],dtype=np.uint8);cuda=np.array([3,4],dtype=np.uint8)
    sampler=tmp_path/'sampler.npz'
    arrays=dict(latent_noise=noise,conditioning_times=times,torch_cpu_rng=cpu,cuda_rng_0=cuda)
    if not energy:arrays['recovery_observation_times']=observations
    np.savez_compressed(sampler,**arrays)
    generation=tmp_path/'generation.npz';arrays={'generated':generated}
    if energy:
        arrays['conditioning_input']=np.concatenate([generated,observations[:,:,None]],axis=2)
    np.savez_compressed(generation,**arrays)
    receipt=dict(path=str(sampler),sha256=replay.sha(sampler),native_generator_training_mode=not release)
    if not energy:receipt.update(native_recovery_training_mode=True,native_grad_enabled=False)
    row=dict(native_joint_step=0 if release else 500,sampler_input=receipt,artifact_path=str(generation),artifact_sha256=replay.sha(generation))
    rng=[]
    torch=SimpleNamespace(load=lambda path,map_location:states[path],
        isfinite=lambda v:Tensor(np.isfinite(v.values)),equal=lambda a,b:np.array_equal(a.values,b.values),
        from_numpy=Tensor,set_rng_state=lambda state:rng.append(('cpu',state.values.copy())),
        set_grad_enabled=lambda grad:nullcontext(),cuda=SimpleNamespace(device_count=lambda:1,
            set_rng_state_all=lambda values:rng.append(('cuda',values[0].values.copy())),empty_cache=lambda:None))
    return row,result,expected,namespace,torch,calls,rng


@pytest.mark.parametrize('energy,release',[(False,False),(True,False),(True,True)])
def test_complete_native_network_load_rng_modes_and_unchanged_full_sampler_recovery(tmp_path,energy,release):
    row,result,b,ns,torch,calls,rng=generation_fixture(tmp_path,energy,release)
    folder=tmp_path/'replay';folder.mkdir()
    report=replay.replay_generation(row,result,b,ns,folder,np,torch)
    assert calls[-2:]==['original_full_sampler','original_full_recovery']
    assert len([v for v in calls if v.startswith('load_')])==(2 if release else 4)
    assert rng[0][0]=='cpu' and np.array_equal(rng[0][1],[1,2])
    assert rng[1][0]=='cuda' and np.array_equal(rng[1][1],[3,4])
    assert ns['generator'].training is (not release) and ns['recovery'].training is (not release)
    assert report['full_prediction_bitwise_equal'] and report['model_parameters_unchanged']
    assert report['numerical_optimizer_updates']==report['additional_generator_training_seeds']==0
    assert report['full_generated_shape']==[7,24,28 if energy else 6]
    assert len(report['full_native_network_state_loads'])==(2 if release else 4)


def test_original_full_generation_cannot_be_replaced_with_a_single_float_bit_change(tmp_path):
    row,result,b,ns,torch,calls,rng=generation_fixture(tmp_path)
    path=Path(row['artifact_path'])
    with np.load(path) as z:changed=z['generated']
    changed[0,0,0]=np.nextafter(changed[0,0,0],np.float32(1))
    np.savez_compressed(path,generated=changed);row['artifact_sha256']=replay.sha(path)
    folder=tmp_path/'replay';folder.mkdir()
    with pytest.raises(ValueError,match='bitwise'):replay.replay_generation(row,result,b,ns,folder,np,torch)


def full_parent_review_fixture(tmp_path,monkeypatch):
    row,result,b,ns,torch,_,_=generation_fixture(tmp_path)
    ns['run_model'].__name__='run_latent_ctfp_model5_3'
    with np.load(row['artifact_path']) as saved:truth=np.array(saved['generated'],copy=True)
    np.savez_compressed(row['artifact_path'],real=truth,generated=truth)
    row['artifact_sha256']=replay.sha(row['artifact_path'])
    audit=tmp_path/'audited_truth.npz';np.savez_compressed(audit,truth=truth.astype(np.float64))
    native_path=tmp_path/'native_result.json';native_path.write_text('{}')
    native_receipt=tmp_path/'native_receipt.json'
    native_receipt.write_text(replay.json.dumps(dict(exit_code=0,resource_guard_aborted=False,full_native_result_verified=True)))
    route=dict(dataset='stock',canonical_job=dict(id='fixture-never-benchmark'),result_path=str(native_path),
        native_completion_receipt=str(native_receipt),full_generation_shape=[7,24,6],
        original_truth_audit=dict(artifact_path=str(audit),artifact_sha256=replay.sha(audit),full_shape=[7,24,6]))
    config_path=tmp_path/'config.json';config_path.write_text('{}')
    config=dict(runtime_configuration=str(config_path),original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273))
    binding=tmp_path/'binding.json'
    frozen=dict(route=route,configuration_sha256=replay.sha(config_path),result_sha256=replay.sha(native_path),
        native_receipt_sha256=replay.sha(native_receipt))
    binding.write_text(replay.json.dumps(frozen))
    folder=tmp_path/'generated_replay';folder.mkdir()
    generation=replay.replay_generation(row,result,b,ns,folder,np,torch)
    evaluators=[];metrics=[]
    for i,role in enumerate(['discriminative']*10+['predictive']*10):
        source=tmp_path/('original_critic_'+str(i));source.mkdir()
        evaluator,eb,code,tf,_,_=evaluator_fixture(source,role)
        b.update(eb)
        evaluators.append(evaluator)
        metrics.append(replay.replay_evaluator(evaluator,eb,code,tmp_path/('critic_replay_'+str(i)),np,tf))
    b['full_periodic_evaluations']=1
    row['evaluator_training_receipts']=evaluators;result['evaluations']=[row]
    monkeypatch.setattr(replay,'budget',lambda _:b)
    monkeypatch.setattr(replay,'original_result',lambda *_:result)
    report=dict(canonical_job=route['canonical_job'],binding_sha256=replay.sha(binding),
        configuration_sha256=replay.sha(config_path),original_native_result_sha256=frozen['result_sha256'],
        original_scope=config['original_scope'],native_release_checkpoint_evaluation=False,
        author_equivalence_certified=False,strict_TAB_result=False,
        all_original_native_generations_and_evaluators_replayed=True,numerical_optimizer_updates=0,additional_generator_training_seeds=0,
        full_periodic_replays=[dict(native_joint_step=500,generation=generation,evaluators=metrics,
            original_truth=replay.verify_native_truth(row,route,np))])
    receipt=dict(exit_code=0,resource_guard_aborted=False,complete_full_native_replay_passed=True)
    return config,route,binding,report,receipt


def test_parent_reviews_saved_complete_arrays_against_native_sources_and_real_completion_binding(tmp_path,monkeypatch):
    arguments=full_parent_review_fixture(tmp_path,monkeypatch)
    assert replay.check_report(*arguments) is arguments[3]


@pytest.mark.parametrize('bad',['generation_bytes','critic_bytes','critic_score','critic_checkpoint','critic_array_binding',
    'missing_critic','source_result_binding','native_receipt_aborted','scope','equality_claim'])
def test_parent_rejects_changed_native_bindings_incomplete_or_falsely_claimed_full_replays(tmp_path,monkeypatch,bad):
    config,route,binding,report,receipt=full_parent_review_fixture(tmp_path,monkeypatch)
    row=report['full_periodic_replays'][0];metric=row['evaluators'][0]
    if bad=='generation_bytes':
        item=row['generation'];path=Path(item['replay_artifact_path'])
        with np.load(path) as saved:values=np.array(saved['generated'],copy=True)
        values[0,0,0]=np.float32(-0.0);np.savez_compressed(path,generated=values)
        item['replay_artifact_sha256']=replay.sha(path)
    elif bad=='critic_bytes':
        path=Path(metric['replay_artifact_path'])
        with np.load(path) as saved:values={key:np.array(saved[key],copy=True) for key in saved.files}
        values['prediction_0'][0,0]=np.nextafter(values['prediction_0'][0,0],np.float32(1))
        np.savez_compressed(path,**values);metric['replay_artifact_sha256']=replay.sha(path)
    elif bad=='critic_score':metric['score']+=.1
    elif bad=='critic_checkpoint':metric['original_checkpoint_files']=[]
    elif bad=='critic_array_binding':metric['original_array_sha256']='changed'
    elif bad=='missing_critic':row['evaluators'].pop()
    elif bad=='source_result_binding':Path(route['result_path']).write_text('{"changed":true}')
    elif bad=='native_receipt_aborted':Path(route['native_completion_receipt']).write_text(replay.json.dumps(dict(exit_code=0,resource_guard_aborted=True,full_native_result_verified=True)))
    elif bad=='scope':report['original_scope']=dict(papers=1)
    elif bad=='equality_claim':report['author_equivalence_certified']=True
    with pytest.raises(ValueError):replay.check_report(config,route,binding,report,receipt)


def test_registration_preserves_actual_regular_stock_loader_without_invented_interpolation():
    from scripts.flow_matching.register_gtgan_all8_full_native_replay_v1 import verify_loader_record
    paths={'stock':'projects/flow_matching_research/results/2026-10-11/gtgan_native_stock_data_audit_v1/original_stock_loader_audit.json',
        'energy':'projects/flow_matching_research/results/2026-10-11/gtgan_energy_native_audit_v3/native_energy_data_and_model_audit.json'}
    tree=ast.parse((SOURCE/'time_dataset.py').read_text(encoding='utf-8'))
    regular=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TimeDataset_regular')
    assert not any(isinstance(n,ast.Attribute) and n.attr=='train_coeffs' for n in ast.walk(regular))
    for dataset,path in paths.items():
        audit=replay.read(replay.ROOT/path)
        for record in audit['records' if dataset=='stock' else 'cases']:
            verify_loader_record(dataset,record)
            for key in ['full_original_truth_and_time_row_identity','original_seed56789_whole_row_mask_exact','all_observed_values_exact',
                    'full_original_cubic_coefficients_finite']:
                changed=copy.deepcopy(record);changed[key]=not record[key]
                with pytest.raises(ValueError):verify_loader_record(dataset,changed)
