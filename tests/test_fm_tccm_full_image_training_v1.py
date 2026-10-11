import copy
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read
from scripts.flow_matching.tccm_full_image_training_contract_v1 import validate_tccm_job
from scripts.flow_matching.run_flow_mismatching_full_training_v1 import capture_rng,restore_rng,save_checkpoint,train_epoch
from iia_benchmark.models.fm_tccm_image import score_tccm_image,tccm_uniform_objective
from tests.test_fm_flow_mismatching_full_training_v1 import TinyFixture,native_optimizer


def fixture_contract():
    job=read(ROOT/'configs/experiments/fm_flow_mismatching_full_training_jobs/flow_mismatching_full__mvtec__conditional__seed1103.json')
    job['method']='TCCM_controlled_image_domain'
    job['training']['epochs']=50;job['training']['checkpoint_interval']=10
    job['expected_optimizer_steps']=5700
    return job,read(ROOT/job['model_configuration']),read(ROOT/job['optimizer_configuration']),read(ROOT/job['dataset_configuration'])


@pytest.mark.parametrize('field,value',[('epochs',49),('epochs',2000),('batch_size',16),
    ('drop_last',True),('early_stopping',True),('gradient_accumulation',2),('mixed_precision','float32')])
def test_full_fifty_epoch_exception_neither_shortened_nor_replaced_by_default(field,value):
    job,model,optimizer,dataset=fixture_contract()
    assert validate_tccm_job(job,model,optimizer,dataset)==3629
    job['training'][field]=value
    with pytest.raises(ValueError):validate_tccm_job(job,model,optimizer,dataset)


def test_subset_or_small_backbone_or_other_optimizer_rejected():
    original=fixture_contract()
    for field in ('subset','backbone','optimizer','seed'):
        job,model,optimizer,dataset=copy.deepcopy(original)
        if field=='subset':job['expected_training_images_per_epoch']=100
        if field=='backbone':model['parameters']['model_channels']=32
        if field=='optimizer':optimizer['parameters']['amsgrad']=False
        if field=='seed':job['seed']=42
        with pytest.raises(ValueError):validate_tccm_job(job,model,optimizer,dataset)


def test_uniform_time_contraction_matches_independent_clean_input_equation_and_gradient():
    images=torch.arange(24,dtype=torch.float64).reshape(2,3,2,2)/10
    multiplier=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64));seen=[]
    labels=torch.tensor([1,2])
    def velocity(x,time,y):
        seen.append((x.clone(),time.clone(),y.clone()))
        return multiplier*x+time[:,None,None,None]
    actual=tccm_uniform_objective(velocity,images,labels=labels,generator=torch.Generator().manual_seed(11))
    time=torch.rand(2,dtype=torch.float64,generator=torch.Generator().manual_seed(11))
    expected=((.2*images+time[:,None,None,None]+images)**2).mean()
    assert torch.equal(seen[0][0],images) and torch.equal(seen[0][1],time) and torch.equal(seen[0][2],labels)
    assert float(actual.detach())==pytest.approx(float(expected),abs=1e-12)
    actual.backward()
    expected_gradient=(2*(.2*images+time[:,None,None,None]+images)*images).mean()
    assert multiplier.grad==pytest.approx(float(expected_gradient),abs=1e-12)


def test_heatmap_is_unsquared_norm_at_exact_t1_and_perfect_contraction_is_trivial_for_anomalies():
    images=torch.tensor([[[[3.]],[[4.]],[[0.]]],[[[30.]],[[40.]],[[0.]]]])
    times=[]
    def zero(x,time):times.append(time);return torch.zeros_like(x)
    heat,score=score_tccm_image(zero,images)
    assert heat[:,0,0].tolist()==[5.,50.] and score.tolist()==[10.,100.]
    assert torch.equal(times[0],torch.ones(2))
    heat,score=score_tccm_image(lambda x,t:-x,images)
    assert torch.count_nonzero(heat)==0 and torch.count_nonzero(score)==0
    # Analytic overfitting failure mode, not original-dataset experimental performance.


def test_native_amsgrad_dropout_time_and_sampler_resume_match_on_cpu_fixture(tmp_path):
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(2)
    images=np.arange(5*3*4*4,dtype=np.float32).reshape(5,3,4,4)/100
    categories=np.arange(5,dtype=np.int64)%3
    model=TinyFixture(True);optimizer=native_optimizer(model);order=torch.Generator().manual_seed(17)
    kwargs=dict(conditional=True,batch_size=2,device=torch.device('cpu'),bf16=False,
        torch=torch,np=np,objective=tccm_uniform_objective)
    first=train_epoch(model,optimizer,images,categories,order=order,**kwargs)
    assert first['images_visited']==first['unique_images_visited']==5 and first['optimizer_steps']==3
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
