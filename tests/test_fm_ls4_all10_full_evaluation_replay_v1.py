import ast
import copy
from pathlib import Path
import numpy as np
import pytest
import random
from types import SimpleNamespace
from scripts.flow_matching import ls4_full_saved_evaluation_replay_v1 as core
from scripts.flow_matching import run_ls4_all10_full_evaluation_replay_v1 as pipeline


def author_source():
    return (core.ROOT/'experiments/runs/fm_spectral_table3_baseline_acquisition_v1/sources/ls4/original/train_interp_extrap.py').read_text(encoding='utf-8')


def test_entire_author_main_prefix_before_optimization_is_preserved():
    source=author_source();main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    boundary=next(i for i,n in enumerate(main.body) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(ast.parse('not args.test',mode='eval').body))
    actual=core.constructor_ast(source).body[0]
    assert [ast.dump(n) for n in actual.body[:-1]]==[ast.dump(n) for n in main.body[:boundary]]
    assert ast.dump(actual.body[-1])==ast.dump(ast.parse('return model, valloader, optimizer, scheduler').body[0])


@pytest.mark.parametrize('change',['boundary','model','loader','device'])
def test_changed_author_constructor_or_optimization_boundary_is_rejected(change):
    source=author_source()
    before,after={'boundary':('if not args.test:','if args.test:'),'model':('model = VAE(config.model)','model = VAE(config.model.other)'),
        'loader':("valloader =  data_objs['test_dataloader']","valloader = data_objs['train_dataloader']"),
        'device':('model = model.to(device)','model = model.to("cpu")')}[change]
    assert before in source
    with pytest.raises(ValueError):core.constructor_ast(source.replace(before,after))


class Tensor:
    device='cuda:0'
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value
    @property
    def dtype(self):return self.value.dtype
    @property
    def shape(self):return self.value.shape


class Torch:
    @staticmethod
    def is_tensor(value):return isinstance(value,Tensor)


def saved_call(folder,name,args,kwargs,value):
    arrays={};inputs=core.encode(dict(args=args,kwargs=kwargs),arrays,'input',np,Torch)
    outputs=core.encode(value,arrays,'output',np,Torch);path=folder/(name+'.npz')
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(method=name,input_structure=inputs,output_structure=outputs,array_path=str(path),array_sha256=core.sha(path),
        array_shapes={k:list(v.shape) for k,v in arrays.items()}),inputs,outputs,arrays


def test_all_original_method_arrays_are_compared_and_output_differences_remain_failures(tmp_path):
    expected,inputs,outputs,arrays=saved_call(tmp_path,'reconstruct',(Tensor([1.,2.]),),dict(get_full_nll=False),Tensor([2.,3.]))
    assert core.compare_call(expected,'reconstruct',inputs,outputs,arrays,np)['all_complete_outputs_bitwise_equal']
    arrays['output'][0]+=1e-5
    result=core.compare_call(expected,'reconstruct',inputs,outputs,arrays,np)
    assert result['all_complete_outputs_bitwise_equal'] is False
    assert result['original_model_call_count']==1 and result['additional_optimizer_updates']==0


@pytest.mark.parametrize('change',['input','keys','shape','dtype','method','nonfinite'])
def test_partial_changed_or_reordered_native_method_inputs_rejected(tmp_path,change):
    expected,inputs,outputs,arrays=saved_call(tmp_path,'reconstruct',(Tensor([1.,2.]),),{},Tensor([2.,3.]))
    name='reconstruct'
    if change=='input':arrays['input.args.0'][0]=5
    elif change=='keys':arrays.pop('output')
    elif change=='shape':arrays['output']=arrays['output'][:1]
    elif change=='dtype':arrays['output']=arrays['output'].astype(np.float32)
    elif change=='method':name='full_nll'
    elif change=='nonfinite':arrays['output'][0]=np.nan
    with pytest.raises(ValueError):core.compare_call(expected,name,inputs,outputs,arrays,np)


def test_scalar_numerical_output_difference_is_recorded_without_early_acceptance_or_abort(tmp_path):
    expected,inputs,_,arrays=saved_call(tmp_path,'full_nll',(),{},1.)
    outputs=core.encode(1.0001,{},'output',np,Torch)
    assert core.compare_call(expected,'full_nll',inputs,outputs,arrays,np)['all_complete_outputs_bitwise_equal'] is False


def test_signed_zero_output_is_not_bitwise_equal(tmp_path):
    expected,inputs,outputs,arrays=saved_call(tmp_path,'full_nll',(),{},Tensor(np.array(0.)))
    arrays['output']=np.array(-0.)
    assert core.compare_call(expected,'full_nll',inputs,outputs,arrays,np)['all_complete_outputs_bitwise_equal'] is False
    assert not core.descriptor_equal(dict(value=0.),dict(value=-0.))


class Model:
    def __init__(self):self.count=0;self.results=[]
    def reconstruct(self,x,**kwargs):
        self.count+=1;value=Tensor(x.value+np.random.normal(size=x.value.shape));self.results.append(value);return value
    def full_nll(self,x,**kwargs):
        self.count+=1;value=Tensor(np.array(np.random.normal()));self.results.append(value);return value


@pytest.mark.parametrize('parallel',[False,True])
def test_whole_native_call_review_preserves_return_identity_call_count_random_order_and_method_attributes(tmp_path,parallel):
    x=Tensor([1.,2.]);model=Model();np.random.seed(23)
    a=model.reconstruct(x);b=model.full_nll(x)
    records=[saved_call(tmp_path,'reconstruct',(x,),{},a)[0],saved_call(tmp_path,'full_nll',(x,),{},b)[0]]
    after=np.random.get_state();np.random.seed(23);model.count=0;model.results=[]
    target=type('Parallel',(),{'module':model})() if parallel else model
    with core.FullCallReview(target,dict(full_method_calls=records),np,Torch) as review:
        assert model.reconstruct(x) is model.results[-1]
        assert model.full_nll(x) is model.results[-1]
    assert review.finish()['all_original_method_outputs_bitwise_equal']
    assert model.count==2 and 'reconstruct' not in model.__dict__ and 'full_nll' not in model.__dict__
    actual=np.random.get_state();assert actual[0]==after[0] and np.array_equal(actual[1],after[1]) and actual[2:]==after[2:]


def test_review_rejects_missing_original_call_and_restores_methods_after_failure(tmp_path):
    x=Tensor([1.]);model=Model();np.random.seed(23);a=model.reconstruct(x)
    record=saved_call(tmp_path,'reconstruct',(x,),{},a)[0]
    with core.FullCallReview(model,dict(full_method_calls=[record]),np,Torch) as review:pass
    with pytest.raises(ValueError):review.finish()
    with pytest.raises(RuntimeError):
        with core.FullCallReview(model,dict(full_method_calls=[record]),np,Torch):raise RuntimeError('native error')
    assert 'reconstruct' not in model.__dict__ and 'full_nll' not in model.__dict__


def test_complete_saved_parameter_buffer_optimizer_scheduler_rng_and_backend_state_restored(monkeypatch):
    class Generator:
        def __init__(self):self.state=Tensor(np.array([0],dtype=np.uint8))
        def get_rng_state(self):return self.state
        def set_rng_state(self,state):self.state=state
    class RestoredModel:
        def state_dict(self):return self.values
        def load_state_dict(self,state,strict):self.values=state;self.strict=strict
        def train(self,value):self.training=value
    class RestoredState:
        def load_state_dict(self,state):self.loaded=state
    class Loader:
        dataset=range(1600);batch_size=64;num_workers=0;drop_last=False;generator=None
        sampler=SimpleNamespace(generator=None)
        def __len__(self):return 25
    cpu_generator=Generator();cuda_states=[Tensor(np.array([1],dtype=np.uint8))]
    cuda=SimpleNamespace(device_count=lambda:1)
    def set_cuda(values):cuda.states=values
    cuda.states=cuda_states;cuda.get_rng_state_all=lambda:cuda.states;cuda.set_rng_state_all=set_cuda
    flags=dict(cudnn_benchmark=True,cudnn_deterministic=False,cudnn_allow_tf32=True,cuda_matmul_allow_tf32=False,
        deterministic_algorithms_enabled=False)
    backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False,deterministic=True,allow_tf32=False),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
    torch=SimpleNamespace(backends=backends,random=cpu_generator,cuda=cuda,deterministic=True,
        equal=lambda a,b:np.array_equal(a.value,b.value))
    torch.cuda.device_count=cuda.device_count
    torch.use_deterministic_algorithms=lambda value:setattr(torch,'deterministic',value)
    torch.are_deterministic_algorithms_enabled=lambda:torch.deterministic
    model=RestoredModel();optimizer=RestoredState();scheduler=RestoredState()
    monkeypatch.setattr(core,'sha',lambda p:Path(p).name+'_hash')
    np.random.seed(7);random.seed(9);numpy_state=np.random.get_state();python_state=random.getstate()
    snapshot=dict(job=dict(id='original',model_config='model'),config_sha256='model_hash',queue_sha256='queue_hash',
        observation_configuration_sha256='observation_hash',native_epoch=10,evaluation_index=0,optimizer_updates=1100,
        classify=False,extrap=False,model=dict(weight=Tensor([1.,2.]),L=Tensor(np.array([2299],dtype=np.int64))),
        optimizer=dict(native_step=1100),scheduler=dict(native_epoch=10),model_training_mode=True,backend_flags=flags,
        loader_generators={},numpy_rng=numpy_state,python_rng=python_state,
        torch_rng=Tensor(np.array([12],dtype=np.uint8)),cuda_rng=[Tensor(np.array([13],dtype=np.uint8))])
    config=dict(epochs=500,train_patients=6400,test_patients=1600,batch_size=64,variables=41,extrapolation=False)
    np.random.normal();random.random()
    core.restore_snapshot(snapshot,dict(queue_path='queue'),snapshot['job'],config,dict(epoch=10),0,
        model,Loader(),optimizer,scheduler,np,torch,'observation')
    assert model.strict and model.training and model.values['L'].value[0]==2299
    assert optimizer.loaded==snapshot['optimizer'] and scheduler.loaded==snapshot['scheduler']
    assert cpu_generator.state.value[0]==12 and cuda.states[0].value[0]==13
    assert core.backend_flags(torch)==flags
    assert np.array_equal(np.random.get_state()[1],numpy_state[1]) and np.random.get_state()[2:]==numpy_state[2:]
    assert random.getstate()==python_state


def fixture(monkeypatch):
    jobs=[];mapping={}
    for profile,extrap in [('interpolation',False),('extrapolation',True)]:
        for seed in [42,1103,1104,1105,1106]:
            name=profile+str(seed);jobs.append(dict(id=name,model_config=name))
            mapping[name]=dict(epochs=500,train_patients=6400,test_patients=1600,batch_size=32 if extrap else 64,
                variables=41,extrapolation=extrap,original_optim={})
    queue=dict(jobs=jobs,settings=dict(commit=52,emergency=16),resource_lock='original_gpu',python='author')
    observation=dict(original_queue='queue',state_root='original')
    mapping['observation']=observation
    config=dict(source_receipts=[],observation_configuration='observation',original_queue='queue',original_jobs=copy.deepcopy(jobs),
        settings=copy.deepcopy(queue['settings']),resource_lock='original_gpu',worker_python='author',training_state_root='original',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        required_full_native_evaluations=500,required_original_model_calls=37500,required_original_training_updates=750000,
        additional_training_seed_slots=0)
    monkeypatch.setattr(pipeline,'read',lambda p:mapping[Path(p).name])
    monkeypatch.setattr(pipeline,'verify_observation',lambda _:queue)
    return config,queue,mapping


def test_all_ten_profiles_seed_slots_full_evaluations_and_full_method_calls_required(monkeypatch):
    config,queue,_=fixture(monkeypatch);assert pipeline.verify(config)==queue


@pytest.mark.parametrize('change',['jobs','duplicate','guard','lock','python','source_owner_root','scope','evaluations','calls','updates','additional_seed','short_epoch','smaller_batch','ema'])
def test_reduced_scope_budgets_original_model_or_original_resources_rejected(monkeypatch,change):
    config,queue,mapping=fixture(monkeypatch)
    if change=='jobs':config['original_jobs'].pop()
    elif change=='duplicate':queue['jobs'][-1]=copy.deepcopy(queue['jobs'][0]);config['original_jobs']=copy.deepcopy(queue['jobs'])
    elif change=='guard':config['settings']['commit']=20
    elif change=='lock':config['resource_lock']='different'
    elif change=='python':config['worker_python']='different'
    elif change=='source_owner_root':config['training_state_root']='different'
    elif change=='scope':config['original_scope']['papers']=44
    elif change=='evaluations':config['required_full_native_evaluations']=499
    elif change=='calls':config['required_original_model_calls']=37499
    elif change=='updates':config['required_original_training_updates']=749999
    elif change=='additional_seed':config['additional_training_seed_slots']=1
    elif change=='short_epoch':mapping['interpolation42']['epochs']=499
    elif change=='smaller_batch':mapping['interpolation42']['batch_size']=32
    elif change=='ema':mapping['interpolation42']['original_optim']['use_ema']=True
    with pytest.raises(ValueError):pipeline.verify(config)


def replay_report_fixture(monkeypatch):
    model=dict(epochs=500,train_patients=6400,test_patients=1600,batch_size=64,variables=41,extrapolation=False)
    job=dict(id='original',model_config='model',output_directory='native')
    output=dict(kind='tensor',key='output',shape=[1],dtype='float32',device='cuda:0')
    inp=dict(kind='tensor',key='input.args.0',shape=[1],dtype='float32',device='cuda:0')
    metrics=dict(eval_mse=.1,eval_crps=.2,eval_full_nll=.3);original=dict(evaluations=[]);rows=[]
    for index,epoch in enumerate(list(range(10,500,10))+[-1]):
        saved=[];calls=[]
        for method in ['reconstruct','full_nll']*25:
            saved.append(dict(method=method,input_structure=inp,output_structure=output,
                array_shapes={'input.args.0':[1],'output':[1]}))
            calls.append(dict(method=method,all_complete_original_tensor_inputs_equal=True,original_model_call_count=1,
                additional_optimizer_updates=0,original_complete_output_structure=output,replayed_complete_output_structure=output,
                output_structure_and_scalars_bitwise_equal=True,all_complete_outputs_bitwise_equal=True,
                full_tensor_comparisons={k:dict(shape=[1],dtype='float32',bitwise_equal=True,maximum_absolute_difference=0.)
                    for k in ['input.args.0','output']}))
        original['evaluations'].append(dict(metrics,full_native_evaluation_observation=dict(
            sha256='context'+str(index),full_test_batches=25,full_method_calls=saved)))
        rows.append(dict(evaluation_index=index,native_epoch=epoch,original_snapshot_sha256='context'+str(index),
            original_full_test_patients=1600,original_full_test_batches=25,full_original_method_calls=50,
            full_original_method_inputs_and_outputs_compared=True,additional_optimizer_updates=0,
            all_original_model_optimizer_scheduler_rng_and_backend_flags_restored=True,calls=calls,
            all_original_method_outputs_bitwise_equal=True,all_original_metrics_exactly_equal=True,
            original_metrics=copy.deepcopy(metrics),replayed_metrics=copy.deepcopy(metrics)))
    report=dict(job=job,original_result_sha256='result.json_hash',training_receipt_sha256='receipt_hash',
        full_native_evaluations_executed=50,all_original_full_model_calls_executed=2500,records=rows,
        all_full_original_model_inputs_and_outputs_compared=True,additional_optimizer_updates=0,
        additional_training_seed_slots=0,author_equivalence_certified=False,strict_TAB_result=False,
        configuration='configuration.json',configuration_sha256='configuration.json_hash',strict_full_saved_outputs_and_metrics_match=True)
    binding=dict(original_result_sha256='result.json_hash',training_receipt_sha256='receipt_hash')
    monkeypatch.setattr(pipeline,'read',lambda p:model if Path(p).name=='model' else original)
    monkeypatch.setattr(pipeline,'sha',lambda p:Path(p).name+'_hash')
    return report,job,{},binding


def test_parent_requires_and_checks_all_fifty_contexts_and_all_original_full_call_descriptors(monkeypatch):
    assert pipeline.verify_complete_replay(*replay_report_fixture(monkeypatch)) is True


def test_numerical_output_failure_is_preserved_after_all_original_full_calls_have_executed(monkeypatch):
    report,job,config,binding=replay_report_fixture(monkeypatch)
    report['records'][0]['calls'][0]['full_tensor_comparisons']['output']['bitwise_equal']=False
    report['records'][0]['calls'][0]['all_complete_outputs_bitwise_equal']=False
    report['records'][0]['all_original_method_outputs_bitwise_equal']=False
    report['strict_full_saved_outputs_and_metrics_match']=False
    assert pipeline.verify_complete_replay(report,job,config,binding) is False
    assert len(report['records'])==50 and report['all_original_full_model_calls_executed']==2500


@pytest.mark.parametrize('change',['context_count','call_count','missing_call','input_identity','wrong_method','missing_tensor',
    'dtype','scalar','false_match','extra_updates','source_snapshot','source_metric'])
def test_partial_or_falsely_passing_replay_reports_cannot_pass_parent_gate(monkeypatch,change):
    report,job,config,binding=replay_report_fixture(monkeypatch);row=report['records'][0];call=row['calls'][0]
    if change=='context_count':report['records'].pop()
    elif change=='call_count':report['all_original_full_model_calls_executed']=2499
    elif change=='missing_call':row['calls'].pop()
    elif change=='input_identity':call['all_complete_original_tensor_inputs_equal']=False
    elif change=='wrong_method':call['method']='other'
    elif change=='missing_tensor':call['full_tensor_comparisons'].pop('output')
    elif change=='dtype':call['full_tensor_comparisons']['output']['dtype']='float64'
    elif change=='scalar':call['replayed_complete_output_structure']=dict(kind='scalar',value=1.)
    elif change=='false_match':call['full_tensor_comparisons']['output']['bitwise_equal']=False
    elif change=='extra_updates':row['additional_optimizer_updates']=1
    elif change=='source_snapshot':row['original_snapshot_sha256']='other'
    elif change=='source_metric':row['original_metrics']['eval_mse']=9.
    with pytest.raises(ValueError):pipeline.verify_complete_replay(report,job,config,binding)
