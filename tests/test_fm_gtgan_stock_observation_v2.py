"""Observation/guard fixtures are plumbing checks, never native benchmark results."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
import types

import numpy as np
import pytest

from scripts.flow_matching import gtgan_stock_observation_runtime_v2 as observer
from scripts.flow_matching import run_gtgan_stock_observation_dispatch_v2 as dispatch
from scripts.flow_matching import run_gtgan_native_queue_v1 as queue_runner
from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import observe_evaluator


QUEUE = observer.read(observer.ROOT/'configs/experiments/fm_gtgan_native_original_stock_queue.v1.json')
CASES = observer.read(observer.ROOT/QUEUE['model_config'])['cases']
SOURCE = observer.ROOT/'data/public_datasets/flow_matching/gtgan_native_source_v1/original'


def main_only(source):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main']
    return tree


def test_entire_frozen_native_worker_is_preserved_after_removing_only_observations():
    source=Path(observer.original.__file__).read_text(encoding='utf-8')
    tree=observer.observed_main_ast(source)
    function=tree.body[0]
    names={'sampler_inputs','metric_receipts','_observation_budget'}
    removed=[]
    for n in function.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):
            removed.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_attach_observers':
            removed.append(n)
    assert len(removed)==4
    function.body=[n for n in function.body if n not in removed]
    keywords={'full_native_sampler_inputs_observed','full_native_evaluator_trainings_observed',
        'native_evaluator_receipts','sampler_inputs','observer_revision','observation_runtime_configuration_sha256'}
    expansion=0;result_fields=0
    for n in ast.walk(tree):
        if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Name) or n.func.id!='dict':continue
        for k in n.keywords[:]:
            if k.arg is None and isinstance(k.value,ast.Call) and k.value.func.id=='_periodic_observation':
                n.keywords.remove(k);expansion+=1
            elif k.arg in keywords:
                n.keywords.remove(k);result_fields+=1
    assert expansion==1 and result_fields==6
    assert ast.dump(tree)==ast.dump(main_only(source))
    assert callable(observer.compile_worker('fixture-config-sha'))


def test_entire_original_gpu_controller_differs_only_in_worker_entry_and_state_root():
    source=Path(queue_runner.__file__).read_text(encoding='utf-8')
    tree=dispatch.controller_ast(source)
    modules=0;bases=0
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value=='scripts.flow_matching.run_gtgan_stock_observation_dispatch_v2':
            n.value='scripts.flow_matching.run_gtgan_native_job_v1';modules+=1
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in n.targets):
            assert ast.dump(n.value)==ast.dump(ast.parse('ROOT/_observed_state_root',mode='eval').body)
            n.value=ast.parse("ROOT/queue['state_root']",mode='eval').body;bases+=1
    assert modules==1 and bases==1
    assert ast.dump(tree)==ast.dump(main_only(source))


def test_original_full_result_verifier_function_has_identical_code_without_host_only_imports():
    import inspect
    function=observer.compile_native_verifier()
    source=Path(queue_runner.__file__).read_text(encoding='utf-8')
    expected=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='verify_result')
    assert ast.dump(ast.parse(inspect.getsource(function)).body[0])==ast.dump(expected)
    assert 'psutil' not in function.__globals__ and 'full_gpu_memory_slot' not in function.__globals__


def test_all_four_actual_original_commands_budgets_and_model_branch_names():
    assert len(CASES)==len(QUEUE['jobs'])==4
    budgets=[observer.extended_budget(c) for c in CASES]
    assert sum(b['full_periodic_evaluations'] for b in budgets)==29
    assert sum(4*b['full_periodic_evaluations'] for b in budgets)==116
    assert sum(20*b['full_periodic_evaluations'] for b in budgets)==580
    for c,b in zip(CASES,budgets):
        old=observer.full_budget(c)
        assert all(b[k]==v for k,v in old.items())
        assert (b['full_generation_windows'],b['sequence_length'],b['variables'],b['latent_noise_variables'])==(3662,24,6,24)
        assert (b['discriminative_evaluator_steps'],b['predictive_evaluator_steps'])==(2000,5000)
        assert c['original_seed']==7777
        filename='GTGAN_stocks.py' if c['missing_rate']==0 else 'GTGAN_stocks_irregular.py'
        tree=ast.parse((SOURCE/filename).read_text(encoding='utf-8'))
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        calls=sorted([n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id in {'discriminative_score_metrics','predictive_score_metrics','predictive_score_metrics2'}],key=lambda n:n.lineno)
        assert [n.func.id for n in calls]==['discriminative_score_metrics',b['predictive_function']]
        assert b['predictive_output_variables']==(1 if c['missing_rate']==0 else 6)
    assert budgets[0]['native_embedding_steps']==0 and budgets[0]['native_joint_steps']==8500
    assert all(b['native_embedding_steps']==10000 and b['native_joint_steps']==2000 for b in budgets[1:])


@pytest.mark.parametrize('key,value',[('first_epoch',10),('max_steps',10),('batch_size',32),
    ('seq_len',12),('max_steps_metric',1),('solver','euler'),('dims','8-8'),('log_time',4),('input_size',6)])
def test_original_budget_and_solver_cannot_be_reduced(key,value):
    case=copy.deepcopy(CASES[0]);case['parsed_original_arguments'][key]=value
    with pytest.raises(ValueError):observer.extended_budget(case)


def fixture_budget(rate=0):
    case=next(c for c in CASES if c['missing_rate']==rate)
    return dict(observer.extended_budget(case),full_generation_windows=7)


class Tensor:
    def __init__(self,array):self.array=np.asarray(array)
    @property
    def shape(self):return self.array.shape
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array


def fixture_torch():
    cpu=np.array([1,2,3],dtype=np.uint8);cuda=np.array([4,5,6],dtype=np.uint8)
    return types.SimpleNamespace(get_rng_state=lambda:Tensor(cpu),is_grad_enabled=lambda:False,
        cuda=types.SimpleNamespace(get_rng_state_all=lambda:[Tensor(cuda)])),cpu,cuda


@pytest.mark.parametrize('full',[True,False])
def test_sampler_delegates_identical_arguments_return_and_rng_without_random_draws(tmp_path,full):
    expected=fixture_budget();torch,cpu,cuda=fixture_torch();calls=[];token=object()
    def native(*args,**kwargs):calls.append((args,kwargs));return token
    module=types.SimpleNamespace(run_model=native,discriminative_score_metrics=lambda:None,predictive_score_metrics2=lambda:None)
    inputs=[];receipts=[]
    observer.attach_observers(module,tmp_path,expected,QUEUE,np,torch,inputs,receipts)
    n=7 if full else 3
    noise=Tensor(np.arange(n*24*24).reshape(n,24,24));times=Tensor(np.zeros((n,24,1)))
    obs=Tensor(np.zeros((n,24)));recovery=types.SimpleNamespace(training=True);generator=types.SimpleNamespace(training=True)
    args=(object(),generator,noise,times,object())
    before=np.random.get_state()
    assert module.run_model(*args,z=True,fixture_flag=token) is token
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert calls==[(args,{'z':True,'fixture_flag':token})] and receipts==[]
    assert np.array_equal(cpu,[1,2,3]) and np.array_equal(cuda,[4,5,6])
    assert len(inputs)==int(full)
    if full:
        with np.load(inputs[0]['path']) as saved:
            assert np.array_equal(saved['latent_noise'],noise.array) and np.array_equal(saved['conditioning_times'],times.array)
            assert np.array_equal(saved['recovery_observation_times'],obs.array)
            assert np.array_equal(saved['torch_cpu_rng'],cpu) and np.array_equal(saved['cuda_rng_0'],cuda)
        assert inputs[0]['native_generator_training_mode'] and inputs[0]['native_recovery_training_mode']
        assert inputs[0]['native_grad_enabled'] is False


@pytest.mark.parametrize('bad',['noise','times','obs'])
def test_full_sampler_incomplete_inputs_fail_before_original_sampler(tmp_path,bad):
    torch,_,_=fixture_torch();calls=[]
    module=types.SimpleNamespace(run_model=lambda *a,**k:calls.append(1),discriminative_score_metrics=lambda:None,predictive_score_metrics2=lambda:None)
    observer.attach_observers(module,tmp_path,fixture_budget(),QUEUE,np,torch,[],[])
    noise=Tensor(np.zeros((7,24,6 if bad=='noise' else 24)))
    times=Tensor(np.zeros((7,23 if bad=='times' else 24,1)))
    obs=Tensor(np.zeros((7,23 if bad=='obs' else 24)));recovery=types.SimpleNamespace(training=True)
    with pytest.raises(ValueError):module.run_model(None,types.SimpleNamespace(training=True),noise,times,None,z=True)
    assert calls==[]


@pytest.mark.parametrize('role,steps',[('discriminative',2000),('predictive',5000)])
def test_full_evaluator_call_is_observed_and_original_y_targets_saved_from_caller(tmp_path,monkeypatch,role,steps):
    closed=[];solver=object();pred=types.SimpleNamespace(name='prediction:0');fakepred=object()
    x=types.SimpleNamespace(name='myinput_x:0');t=types.SimpleNamespace(name='myinput_t:0')
    # Placeholder keys must be hashable, as actual TF tensors are.
    class Key:
        def __init__(self,name):self.name=name
    x=Key('myinput_x:0');t=Key('myinput_t:0')
    target=np.zeros((7,23,1));prediction=target+.125
    class Session:
        def __init__(self,*a,**k):self.graph=types.SimpleNamespace(as_default=nullcontext)
        def run(self,fetches,feed_dict=None,*a,**k):
            if isinstance(fetches,list) and fetches[0] is solver:return [None,0.]
            if isinstance(fetches,list):return [np.ones((2,1))*.8,np.ones((2,1))*.2]
            return prediction
        def close(self):closed.append(True)
    class Saver:
        def __init__(self,**kwargs):assert len(kwargs['var_list'])==8
        def save(self,session,prefix,write_meta_graph):
            assert write_meta_graph
            for suffix in ['.meta','.index','.data-00000-of-00001']:Path(prefix+suffix).write_bytes(b'fixture-only-not-native-weights')
    fake=types.SimpleNamespace(compat=types.SimpleNamespace(v1=types.SimpleNamespace(Session=Session,
        ConfigProto=lambda **kw:kw,global_variables=lambda:list(range(8)),train=types.SimpleNamespace(Saver=Saver))))
    import sys
    monkeypatch.setitem(sys.modules,'tensorflow',fake)
    def native():
        d_solver=p_solver=solver;y_pred_real=y_pred=pred;y_pred_fake=fakepred;Y_mb=target
        sess=fake.compat.v1.Session()
        for _ in range(steps):sess.run([solver,object()],feed_dict={x:np.zeros((7,23,5)),t:np.ones(7)*23})
        if role=='discriminative':sess.run([y_pred_real,y_pred_fake],feed_dict={x:np.zeros((2,24,6)),t:np.ones(2)*24});return .5
        sess.run(y_pred,feed_dict={x:np.zeros((7,23,5)),t:np.ones(7)*23});return .125
    native.native_function_name='fixture_'+role
    score,receipt=observe_evaluator(native,role,tmp_path/'metric',steps)
    assert score==(.5 if role=='discriminative' else .125) and receipt['optimizer_updates']==steps
    assert receipt['independent_checkpoint_replay_complete'] is False and receipt['evaluator_training_instances']==1
    assert len(receipt['checkpoint_files'])==3 and closed==[True] and fake.compat.v1.Session is Session
    with np.load(receipt['array_path']) as arrays:
        assert 'myinput_x:0' in arrays and 'myinput_t:0' in arrays
        if role=='predictive':assert np.array_equal(arrays['original_reference'],target) and 'myinput_y:0' not in arrays


def periodic_fixture(tmp_path,rate=0):
    expected=fixture_budget(rate);n=expected['full_generation_windows'];length=24;variables=6
    sampler_path=tmp_path/'sampler.npz'
    np.savez_compressed(sampler_path,latent_noise=np.zeros((n,length,24)),conditioning_times=np.zeros((n,length,1)),
        recovery_observation_times=np.zeros((n,length)),torch_cpu_rng=np.array([1,2],dtype=np.uint8),cuda_rng_0=np.array([3,4],dtype=np.uint8))
    sampler=dict(path=str(sampler_path),sha256=observer.sha(sampler_path),full_windows=n,
        native_generator_training_mode=True,native_recovery_training_mode=True,native_grad_enabled=False)
    genpath=tmp_path/'generation.npz';np.savez_compressed(genpath,real=np.zeros((n,length,variables)),generated=np.zeros((n,length,variables)))
    receipts=[]
    for index,role in enumerate(['discriminative']*10+['predictive']*10):
        folder=tmp_path/str(index);folder.mkdir();path=folder/'arrays.npz'
        if role=='discriminative':
            m=n-int(.8*n)
            np.savez_compressed(path,prediction_real=np.ones((m,1))*.8,prediction_fake=np.ones((m,1))*.2,
                **{'myinput_x:0':np.zeros((m,length,variables)),'myinput_x_hat:0':np.zeros((m,length,variables)),
                    'myinput_t:0':np.ones(m)*24,'myinput_t_hat:0':np.ones(m)*24})
            score=.5;function='discriminative_score_metrics'
        else:
            out=expected['predictive_output_variables']
            np.savez_compressed(path,prediction=np.ones((n,23,out))*.125,original_reference=np.zeros((n,23,out)),
                **{'myinput_x:0':np.zeros((n,23,variables if out==variables else variables-1)),'myinput_t:0':np.ones(n)*23})
            score=.125;function=expected['predictive_function']
        checkpoints=[]
        for suffix in ['.meta','.index','.data-00000-of-00001']:
            p=folder/('native_evaluator'+suffix);p.write_bytes(b'fixture-only')
            checkpoints.append(dict(path=str(p),sha256=observer.sha(p)))
        receipts.append(dict(role=role,evaluator_function=function,optimizer_updates=expected[role+'_evaluator_steps'],
            expected_optimizer_updates=expected[role+'_evaluator_steps'],full_saved_prediction_score_check_passed=True,
            trained_variable_count=8,evaluator_training_instances=1,additional_generator_training_seeds=0,
            array_path=str(path),array_sha256=observer.sha(path),checkpoint_files=checkpoints,
            original_returned_score=score,full_saved_predictions_recomputed_score=score,independent_checkpoint_replay_complete=False))
    row=dict(sampler_input=sampler,evaluator_training_receipts=receipts,artifact_path=str(genpath),artifact_sha256=observer.sha(genpath),
        discriminative_repeats=[.5]*10,predictive_repeats=[.125]*10,
        original_metrics=dict(discriminative=.5,discriminative_std=0.,predictive=.125,predictive_std=0.))
    return expected,row


@pytest.mark.parametrize('rate',[0,.3,.5,.7])
def test_complete_saved_artifacts_and_original_score_records_are_checked(tmp_path,rate):
    budget,row=periodic_fixture(tmp_path,rate)
    assert observer.periodic_observation(0,[row['sampler_input']],row['evaluator_training_receipts'],budget)=={
        'sampler_input':row['sampler_input'],'evaluator_training_receipts':row['evaluator_training_receipts']}
    observer.verify_periodic_artifacts(row,budget)
    assert all(r['independent_checkpoint_replay_complete'] is False for r in row['evaluator_training_receipts'])


@pytest.mark.parametrize('bad',['one_missing','order','updates','function','score','means','training_seed',
    'checkpoint_missing','checkpoint_tamper','array_tamper','sampler_tamper','mode','full_prediction_coverage'])
def test_incomplete_or_changed_native_evidence_cannot_be_accepted(tmp_path,bad):
    budget,row=periodic_fixture(tmp_path);e=row['evaluator_training_receipts'][0]
    if bad=='one_missing':row['evaluator_training_receipts'].pop()
    elif bad=='order':row['evaluator_training_receipts'].reverse()
    elif bad=='updates':e['optimizer_updates']=1999
    elif bad=='function':row['evaluator_training_receipts'][10]['evaluator_function']='predictive_score_metrics'
    elif bad=='score':e['original_returned_score']=.7
    elif bad=='means':row['original_metrics']['predictive']=.2
    elif bad=='training_seed':e['additional_generator_training_seeds']=1
    elif bad=='checkpoint_missing':e['checkpoint_files'].pop()
    elif bad=='checkpoint_tamper':Path(e['checkpoint_files'][0]['path']).write_bytes(b'changed')
    elif bad=='array_tamper':Path(e['array_path']).write_bytes(b'changed')
    elif bad=='sampler_tamper':Path(row['sampler_input']['path']).write_bytes(b'changed')
    elif bad=='mode':del row['sampler_input']['native_recovery_training_mode']
    else:
        with np.load(e['array_path']) as saved:arrays={key:saved[key] for key in saved.files}
        arrays['myinput_x:0']=arrays['myinput_x:0'][:1]
        np.savez_compressed(e['array_path'],**arrays);e['array_sha256']=observer.sha(e['array_path'])
    with pytest.raises(ValueError):
        observer.periodic_observation(0,[row['sampler_input']],row['evaluator_training_receipts'],budget)
        observer.verify_periodic_artifacts(row,budget)


@pytest.mark.parametrize('state,descendants,fresh,allowed',[
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':False},[],True,True),
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':False},[{'name':'conhost.exe','exe':'C:/Windows/System32/conhost.exe'}],True,True),
    ({'state':'waiting_full_gpu_resources','active_pid':44,'shared_lock_owned':False},[],True,False),
    ({'state':'running_full_original_native_command','active_pid':None,'shared_lock_owned':False},[],True,False),
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':False},[{'name':'python.exe','exe':'C:/python.exe'}],True,False),
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':False},[{'name':'conhost.exe','exe':'C:/other/conhost.exe'}],True,False),
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':False},[],False,False),
    ({'state':'waiting_full_gpu_resources','active_pid':None,'shared_lock_owned':True},[],True,False)])
def test_only_exact_idle_gpu_parent_without_native_children_or_attempts_may_handoff(state,descendants,fresh,allowed):
    assert dispatch.idle_parent_phase(state,descendants,fresh) is allowed


@pytest.mark.parametrize('artifact',['output','receipts','sampler'])
def test_any_started_or_preserved_stock_attempt_prevents_idle_handoff(tmp_path,artifact):
    runtime=tmp_path/'runtime';runtime.mkdir();output=tmp_path/'output'
    queue={'jobs':[{'runtime_root':str(runtime),'output_directory':str(output)}]}
    assert dispatch.all_attempts_unstarted(queue)
    if artifact=='output':output.mkdir()
    elif artifact=='receipts':(runtime/'native_evaluator_receipts').mkdir()
    else:(runtime/'full_sampler_inputs_1.npz').write_bytes(b'preserved')
    assert not dispatch.all_attempts_unstarted(queue)


def config_fixture():
    return dict(source_receipts=[],original_queue=QUEUE['queue_path'],settings=copy.deepcopy(QUEUE['settings']),
        resource_lock=QUEUE['resource_lock'],cpu_threads=2,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        required_sampler_snapshots=29,required_trained_evaluator_receipts=580,required_native_network_checkpoints=116,
        native_latent_noise_shape=[3662,24,24],native_real_and_generated_shape=[3662,24,6])


@pytest.mark.parametrize('key,value',[('required_sampler_snapshots',1),('required_trained_evaluator_receipts',20),
    ('required_native_network_checkpoints',4),('resource_lock','separate.lock'),('cpu_threads',1),
    ('native_latent_noise_shape',[3662,24,6]),('native_real_and_generated_shape',[128,24,6]),
    ('original_scope',{'papers':1,'method_baseline_ablation_records':1,'original_data_task_records':1})])
def test_full_scope_or_observation_obligations_cannot_be_reduced(key,value):
    config=config_fixture();config[key]=value
    with pytest.raises(ValueError):dispatch.verify(config)


@pytest.mark.parametrize('key,value',[('minimum_commit_headroom_bytes',1),('minimum_gpu_free_bytes',1),
    ('minimum_available_memory_bytes',1),('emergency_commit_headroom_bytes',1),('emergency_gpu_free_bytes',1),
    ('gpu_uuid','different'),('controller_yield_seconds',0)])
def test_original_gpu_admission_emergency_and_scheduler_settings_unchanged(key,value):
    config=config_fixture();config['settings'][key]=value
    with pytest.raises(ValueError):dispatch.verify(config)


def test_full_frozen_stock_registration_contract_passes():
    assert dispatch.verify(config_fixture())==QUEUE
