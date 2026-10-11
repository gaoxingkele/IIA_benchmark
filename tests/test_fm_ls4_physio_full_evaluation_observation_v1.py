"""Full LS4 observation plumbing and unchanged author AST; no performance evidence."""
import ast
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import ls4_physio_full_evaluation_observation_v1 as observation


def test_entire_original_worker_is_restored_after_removing_only_observation_bindings():
    path=observation.ROOT/'scripts/flow_matching/run_ls4_native_physionet_v1.py'
    source=path.read_text(encoding='utf-8')
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_job')
    adapted=observation.worker_ast(source).body[0]
    evaluate=next(n for n in adapted.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    restored=[];removed=0
    for node in evaluate.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='observation':removed+=1;continue
        if isinstance(node,ast.With) and isinstance(node.items[0].context_expr,ast.Name) and node.items[0].context_expr.id=='observation':
            assert len(node.body)==1;restored.extend(node.body);removed+=1;continue
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].slice,ast.Constant) and node.targets[0].slice.value=='full_native_evaluation_observation':removed+=1;continue
        restored.append(node)
    evaluate.body=restored
    adapted.body=[n for n in adapted.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript)
        and isinstance(n.targets[0].slice,ast.Constant) and n.targets[0].slice.value=='full_native_evaluation_observation_configuration_sha256')]
    assert removed==3 and ast.dump(adapted)==ast.dump(original)
    compile(observation.worker_ast(source),str(path),'exec')


def test_missing_or_altered_original_native_evaluation_invocation_cannot_be_observed():
    source=(observation.ROOT/'scripts/flow_matching/run_ls4_native_physionet_v1.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError):observation.worker_ast(source.replace('value = original_eval(model, device, epoch, observed, best_mse, log_dir, classify=classify, extrap=extrap)',
        'value = original_eval(model, device, epoch, loader, best_mse, log_dir, classify=classify, extrap=extrap)'))


def test_all_ten_original_profiles_seeds_and_full_evaluation_call_obligations():
    queue=observation.read(observation.ROOT/'configs/experiments/fm_ls4_native_physionet_queue.v3.json')
    assert len(queue['jobs'])==10
    calls=0;updates=0
    for j in queue['jobs']:
        config=observation.read(observation.ROOT/j['model_config']);b=observation.budget(config)
        assert config['seed'] in [42,1103,1104,1105,1106]
        assert len(observation.periodic_epochs(500,10)+[-1])==50
        calls+=50*2*b['test_batches'];updates+=b['optimizer_updates']
    assert calls==37500 and updates==750000


class Tensor:
    def __init__(self,value):self.value=np.asarray(value);self.device='cuda:0'
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


def fixture(tmp_path,extrap=False):
    calls=[]
    config=dict(epochs=500,train_patients=6400,test_patients=1600,variables=41,
        batch_size=32 if extrap else 64,extrapolation=extrap)
    config_path=tmp_path/'model.json';config_path.write_text('{}')
    queue_path=tmp_path/'queue.json';queue_path.write_text('{}')
    observation_path=tmp_path/'observation.json';observation_path.write_text('{}')
    job=dict(id='fixture-never-benchmark',model_config=str(config_path))
    queue=dict(queue_path=str(queue_path))
    class Model:
        training=True
        def state_dict(self):return dict(weight=np.arange(3,dtype=np.float32))
        def reconstruct(self,x,t,t_pred=None,masks=None,get_full_nll=False):
            calls.append(dict(method='reconstruct',args=[x,t,t_pred],masks=masks,get_full_nll=get_full_nll))
            return x
        def full_nll(self,x,t,masks=None):
            calls.append(dict(method='full_nll',args=[x,t],masks=masks));return Tensor(np.float32(1))
    model=Model()
    count=50 if extrap else 25
    loader=SimpleNamespace(dataset=range(1600),batch_size=config['batch_size'],num_workers=0,drop_last=False,
        sampler=SimpleNamespace(generator=None),generator=None)
    class Loader(type(loader)):
        def __len__(self):return count
    loader=Loader(**vars(loader))
    state=SimpleNamespace(state_dict=lambda:dict(full_native_state='fixture-only'))
    torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,Tensor),save=lambda value,stream:pickle.dump(value,stream),
        random=SimpleNamespace(get_rng_state=lambda:np.array([1,2],dtype=np.uint8)),
        cuda=SimpleNamespace(get_rng_state_all=lambda:[np.array([3,4],dtype=np.uint8)]),is_grad_enabled=lambda:True,
        are_deterministic_algorithms_enabled=lambda:False,
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True,deterministic=False,allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    observer=observation.FullEvaluationObservation(str(observation_path),queue,job,config,tmp_path/'native',model,loader,
        state,state,10,0,float('inf'),False,extrap,1000,np,torch)
    return observer,model,loader,calls,torch


@pytest.mark.parametrize('extrap',[False,True])
def test_full_model_optimizer_scheduler_rng_and_all_native_method_inputs_outputs_preserved(tmp_path,extrap):
    previous_np=np.random.get_state();previous_python=observation.random.getstate()
    observer,model,loader,calls,torch=fixture(tmp_path,extrap)
    assert np.array_equal(np.random.get_state()[1],previous_np[1]) and observation.random.getstate()==previous_python
    x=Tensor(np.arange(loader.batch_size*2*41,dtype=np.float32).reshape(loader.batch_size,2,41))
    t=Tensor(np.arange(2,dtype=np.float32));mask=Tensor(np.ones_like(x.value));original=model.reconstruct
    with observer:
        for i in range(len(loader)):
            assert model.reconstruct(x,t,None,masks=mask,get_full_nll=False) is x
            model.full_nll(x,t,masks=mask)
    assert model.reconstruct==original and 'reconstruct' not in model.__dict__ and 'full_nll' not in model.__dict__
    receipt=observer.finish()
    assert len(calls)==2*len(loader) and len(receipt['full_method_calls'])==len(calls)
    assert receipt['full_observation_complete'] and receipt['numerical_optimizer_updates_from_observation']==0
    snapshot=pickle.loads(Path(receipt['path']).read_bytes())
    assert snapshot['model_training_mode'] is True and snapshot['grad_enabled'] is True
    assert snapshot['backend_flags']==observation.backend_flags(torch)
    assert set(snapshot)>=set(['model','optimizer','scheduler','torch_rng','cuda_rng','numpy_rng','python_rng','loader_generators'])
    assert snapshot['best_mse']==float('inf') and snapshot['optimizer_updates']==1000
    for row in receipt['full_method_calls']:
        assert observation.sha(row['array_path'])==row['array_sha256']
        with np.load(row['array_path']) as arrays:
            assert np.array_equal(arrays['input.args.0'],x.value)
            assert np.array_equal(arrays['input.kwargs.masks'],mask.value)
            assert arrays['output'].shape==(x.value.shape if row['method']=='reconstruct' else ())


def test_observation_restores_both_model_methods_after_native_failure_without_success_receipt(tmp_path):
    observer,model,loader,calls,_=fixture(tmp_path)
    original=model.reconstruct
    with pytest.raises(RuntimeError):
        with observer:raise RuntimeError('original method fixture failure')
    assert model.reconstruct==original and 'reconstruct' not in model.__dict__ and 'full_nll' not in model.__dict__
    assert (observer.folder/'failed_observation.json').exists()
    assert not (observer.folder/'complete_original_evaluation_observation.json').exists()


def test_missing_whole_original_method_call_cannot_be_marked_complete(tmp_path):
    observer,model,loader,calls,_=fixture(tmp_path)
    with observer:model.reconstruct(Tensor(np.ones((64,2,41),dtype=np.float32)),Tensor(np.arange(2,dtype=np.float32)))
    with pytest.raises(ValueError):observer.finish()


def test_full_tensor_structure_copy_rejects_unsupported_silent_omission():
    arrays={};torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,Tensor))
    value=Tensor(np.array([1,2],dtype=np.float32))
    structure=observation.encode(dict(args=(value,None),kwargs=dict(mask=[value])),arrays,'input',np,torch)
    assert structure['items']['args']['kind']=='tuple'
    value.value[0]=99
    assert arrays['input.args.0'][0]==1
    with pytest.raises(ValueError):observation.encode(set([1]),{},'bad',np,torch)


def test_entire_original_controller_is_preserved_except_observation_entry_state_and_config():
    from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import controller_ast
    path=observation.ROOT/'scripts/flow_matching/run_ls4_native_physionet_queue_v3.py'
    source=path.read_text(encoding='utf-8')
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    adapted=controller_ast(source).body[0]
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1':node.value='scripts.flow_matching.run_ls4_native_physionet_v1'
            return node
    adapted=Reverse().visit(adapted)
    additions=[ast.parse('queue = dict(queue,state_root=observation_state_root)').body[0],
        ast.parse('state["observation_configuration_sha256"] = sha(ROOT/observation_configuration)').body[0]]
    before=len(adapted.body)
    adapted.body=[n for n in adapted.body if all(ast.dump(n)!=ast.dump(a) for a in additions)]
    assert before-len(adapted.body)==2
    launch=next(n for n in adapted.body if isinstance(n,ast.FunctionDef) and n.name=='launch')
    extension=ast.parse('command.extend(["--runtime-config",observation_configuration])').body[0]
    before=len(launch.body);launch.body=[n for n in launch.body if ast.dump(n)!=ast.dump(extension)]
    assert before-len(launch.body)==1 and ast.dump(adapted)==ast.dump(original)
    compile(controller_ast(source),str(path),'exec')


def test_original_random_sampling_order_and_return_identity_survive_method_observation(tmp_path):
    observer,model,_,calls,_=fixture(tmp_path)
    def sampled(x,t,**kw):
        calls.append('actual_native_random_method');return Tensor(np.random.standard_normal((64,2,41)).astype(np.float32))
    model.reconstruct=sampled
    inputs=(Tensor(np.zeros((64,2,41),dtype=np.float32)),Tensor(np.arange(2,dtype=np.float32)))
    state=np.random.get_state();expected=sampled(*inputs);after=np.random.get_state()
    np.random.set_state(state)
    with observer:actual=model.reconstruct(*inputs)
    assert actual.value.tobytes()==expected.value.tobytes()
    assert np.array_equal(np.random.get_state()[1],after[1]) and np.random.get_state()[2:]==after[2:]
    assert model.reconstruct is sampled and calls==['actual_native_random_method']*2


def dispatch_config():
    queue=observation.read(observation.ROOT/'configs/experiments/fm_ls4_native_physionet_queue.v3.json')
    return dict(source_receipts=[],original_queue=queue['queue_path'],full_original_jobs=queue['jobs'],
        original_settings=queue['settings'],original_gpu_mutex=queue['resource_lock'],additional_training_seed_slots=0,
        required_full_native_evaluations=500,required_full_native_reconstruction_and_nll_calls=37500,
        required_original_training_optimizer_updates=750000,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273))


def test_exact_complete_observation_configuration_requires_all_original_jobs_and_guards():
    from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import verify
    config=dispatch_config();assert len(verify(config)['jobs'])==10


@pytest.mark.parametrize('key,value',[
    ('required_full_native_evaluations',499),('required_full_native_reconstruction_and_nll_calls',37499),
    ('required_original_training_optimizer_updates',749999),('additional_training_seed_slots',1),
    ('original_scope',dict(papers=1)),('original_gpu_mutex','different.lock')])
def test_reduced_or_changed_full_native_observation_scope_rejected(key,value):
    from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import verify
    config=dispatch_config();config[key]=value
    with pytest.raises(ValueError):verify(config)


@pytest.mark.parametrize('change',['drop_job','lower_commit','lower_emergency'])
def test_original_native_job_count_and_memory_guards_cannot_be_reduced(change):
    from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import verify
    config=copy.deepcopy(dispatch_config())
    if change=='drop_job':config['full_original_jobs'].pop()
    else:config['original_settings']['minimum_commit_headroom_bytes' if change=='lower_commit' else 'emergency_commit_headroom_bytes']-=1
    with pytest.raises(ValueError):verify(config)


@pytest.mark.parametrize('bad',[None,'worker','exe','argument','late_creation','command_path'])
def test_idle_handoff_excludes_only_the_actual_original_system_console_host(bad,monkeypatch):
    from scripts.flow_matching.register_ls4_physio_full_observation_dispatch_v1 import verified_idle_console_children
    executable='C:\\Windows\\System32\\conhost.exe';command=['\\??\\'+executable,'0x4']
    name='conhost.exe';created=100.01
    monkeypatch.setenv('SystemRoot','C:\\Windows')
    if bad=='worker':name='python.exe'
    elif bad=='exe':executable='D:\\unverified\\conhost.exe'
    elif bad=='argument':command[1]='different'
    elif bad=='late_creation':created=101.01
    elif bad=='command_path':command[0]='D:\\unverified\\conhost.exe'
    child=SimpleNamespace(pid=123,name=lambda:name,exe=lambda:executable,cmdline=lambda:command,create_time=lambda:created)
    parent=SimpleNamespace(children=lambda recursive:[child],create_time=lambda:100)
    if bad is None:assert verified_idle_console_children(parent)[0]['verified_original_windows_console_host']
    else:
        with pytest.raises(ValueError):verified_idle_console_children(parent)
