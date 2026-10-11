import copy
from functools import partial
import random
import numpy as np
import pytest
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read
from scripts.flow_matching.wt_full_image_training_contract_v1 import WT_PARAMETERS,validate_wt_job
from scripts.flow_matching.run_flow_mismatching_full_training_v1 import capture_rng,restore_rng,save_checkpoint,train_epoch
from iia_benchmark.models.fm_reflect_wt_native import load_verified_wt_flow
from iia_benchmark.models.fm_wt_image_training import wt_uniform_objective
from tests.test_fm_flow_mismatching_full_training_v1 import TinyFixture,native_optimizer


def fixture_contract():
    job=read(ROOT/'configs/experiments/fm_flow_mismatching_full_training_jobs/flow_mismatching_full__mvtec__conditional__seed1103.json')
    job['method']='WT_Flow_controlled_image_domain'
    job['native_components_configuration']='configs/models/fm_reflect_wt_native_components.v1.json'
    job['original_WT_best_feature_setting_reproduced']=False
    model=read(ROOT/job['model_configuration']);model['method_parameters']=copy.deepcopy(WT_PARAMETERS)
    native=read(ROOT/job['native_components_configuration'])['wt']
    model['native_source']={k:native[k] for k in ('source_path','expected_sha256')}
    return job,model,read(ROOT/job['optimizer_configuration']),read(ROOT/job['dataset_configuration'])


def native_core():
    model=fixture_contract()[1]
    return load_verified_wt_flow(**model['native_source'],asset_root=ROOT)


@pytest.fixture(autouse=True)
def restore_state():
    threads=torch.get_num_threads();torch.set_num_threads(2)
    python=random.getstate();numpy=np.random.get_state();state=torch.get_rng_state()
    yield
    random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(state);torch.set_num_threads(threads)


@pytest.mark.parametrize('field,value',[('epochs',50),('epochs',1999),('batch_size',16),
    ('drop_last',True),('early_stopping',True),('gradient_accumulation',2),('mixed_precision','float32')])
def test_full2000epoch_same_backbone_budget_and_tail_cannot_be_reduced(field,value):
    job,model,optimizer,dataset=fixture_contract()
    assert validate_wt_job(job,model,optimizer,dataset)==3629
    job['training'][field]=value
    with pytest.raises(ValueError):validate_wt_job(job,model,optimizer,dataset)


@pytest.mark.parametrize('kind',['subset','backbone','optimizer','seed','conditioning','epsilon','direction','native_source'])
def test_no_subset_backbone_or_changed_native_statistics(kind):
    job,model,optimizer,dataset=fixture_contract()
    if kind=='subset':job['expected_training_images_per_epoch']=100
    if kind=='backbone':model['parameters']['model_channels']=32
    if kind=='optimizer':optimizer['parameters']['amsgrad']=False
    if kind=='seed':job['seed']=42
    if kind=='conditioning':job['variant']='pooled';model['parameters']['num_classes']=None
    if kind=='epsilon':model['method_parameters']['epsilon']=1e-8
    if kind=='direction':model['method_parameters']['transport_direction']='gaussian_to_normal'
    if kind=='native_source':model['native_source']['expected_sha256']='0'*64
    with pytest.raises(ValueError):validate_wt_job(job,model,optimizer,dataset)


def test_full128rgb_uniform_time_gaussian_and_native_reverse_loss_match_closed_equation():
    images=torch.arange(2*3*128*128,dtype=torch.float64).reshape(2,3,128,128)/1000
    coefficient=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64));seen=[]
    labels=torch.tensor([2,4])
    def velocity(state,time,y):seen.append((state.detach().clone(),time.clone(),y.clone()));return coefficient*state
    actual_generator=torch.Generator().manual_seed(417)
    loss=wt_uniform_objective(velocity,images,native_core=native_core(),epsilon=1e-5,
        labels=labels,generator=actual_generator)
    independent_generator=torch.Generator().manual_seed(417)
    t=torch.rand(2,dtype=torch.float64,generator=independent_generator)
    noise=torch.randn(images.shape,dtype=torch.float64,generator=independent_generator)
    mean=images.mean((1,2,3),keepdim=True)
    mapped=(images-mean)/torch.sqrt((images-mean).square().mean((1,2,3),keepdim=True)+1e-5)
    state=(1-t[:,None,None,None])*mapped+t[:,None,None,None]*noise
    residual=coefficient.detach()*state-(noise-mapped)
    torch.testing.assert_close(loss,residual.square().mean(),rtol=1e-12,atol=1e-12)
    loss.backward();torch.testing.assert_close(coefficient.grad,2*(residual*state).mean(),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(seen[0][0],state,rtol=1e-12,atol=1e-12)
    assert torch.equal(seen[0][1],t) and torch.equal(seen[0][2],labels)
    assert torch.equal(actual_generator.get_state(),independent_generator.get_state())


def test_native_AMSGrad_dropout_Gaussian_time_and_full_tail_resume_exact_on_cpu(tmp_path):
    random.seed(31);np.random.seed(31);torch.manual_seed(31)
    images=np.arange(7*3*4*4,dtype=np.float32).reshape(7,3,4,4)/100
    categories=np.arange(7,dtype=np.int64)%3
    model=TinyFixture(True);optimizer=native_optimizer(model);order=torch.Generator().manual_seed(31)
    objective=partial(wt_uniform_objective,native_core=native_core(),epsilon=1e-5)
    kwargs=dict(conditional=True,batch_size=2,device=torch.device('cpu'),bf16=False,torch=torch,np=np,objective=objective)
    first=train_epoch(model,optimizer,images,categories,order=order,**kwargs)
    assert first['images_visited']==first['unique_images_visited']==7 and first['optimizer_steps']==4
    path=tmp_path/'resume.pt'
    save_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(torch,np,order)),torch)
    expected=train_epoch(model,optimizer,images,categories,order=order,**kwargs)
    weights=copy.deepcopy(model.state_dict());states=copy.deepcopy(optimizer.state_dict())
    restored=TinyFixture(True);resumed=native_optimizer(restored);other=torch.Generator()
    checkpoint=torch.load(path,weights_only=False)
    restored.load_state_dict(checkpoint['model']);resumed.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'],torch,np,other)
    assert train_epoch(restored,resumed,images,categories,order=other,**kwargs)==expected
    for key,value in weights.items():assert torch.equal(value,restored.state_dict()[key])
    for key,values in states['state'].items():
        for name,value in values.items():
            actual=resumed.state_dict()['state'][key][name]
            assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
