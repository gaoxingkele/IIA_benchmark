"""Equation/recipe checks on artificial arrays, never scientific performance."""
import copy
import json
from pathlib import Path

import pytest
import torch

from iia_benchmark.models.fm_image_comparison import (
    reflect_corruption,reflect_objective,reflect_heatmap,tccm_objective,tccm_heatmap,
    wt_affine,wt_flow_objective,wt_flow_heatmap,reconflow_heatmap,
)
from scripts.flow_matching.audit_flow_mismatching_original_protocol_v1 import validate_protocol


@pytest.fixture(scope='module',autouse=True)
def bounded_threads():
    original=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_reflect_corrupts_only_mask_and_retains_clean_target():
    clean=torch.tensor([[[[3.,4.]],[[0.,0.]]]])
    mask=torch.tensor([[[[0.,1.]]]])
    noisy=reflect_corruption(clean,mask,alpha=torch.tensor(1.),beta=torch.tensor(0.),
                            spatial_noise=torch.zeros_like(clean),scalar_noise=torch.tensor(0.))
    torch.testing.assert_close(noisy,torch.tensor([[[[3.,0.]],[[0.,0.]]]]))
    boolean=reflect_corruption(clean,mask.bool(),alpha=torch.tensor(1.),beta=torch.tensor(0.),
                              spatial_noise=torch.zeros_like(clean),scalar_noise=torch.tensor(0.))
    assert torch.equal(noisy,boolean)
    theta=torch.tensor(0.,requires_grad=True)
    loss=reflect_objective(lambda x,t:theta*x,clean,noisy,torch.tensor([.5]))
    assert loss.item()==4
    loss.backward();assert theta.grad.item()<0
    heat=reflect_heatmap(lambda x,t:x,clean)
    torch.testing.assert_close(heat,torch.tensor([[[3.,4.]]]))


def test_tccm_contracts_data_at_all_training_times_and_unsquared_test_t1():
    data=torch.tensor([[[[3.]],[[4.]]],[[[0.]],[[2.]]]])
    time=torch.tensor([.2,.8])
    assert tccm_objective(lambda x,t:-x,data,time).item()==0
    observed=[]
    def zero(x,t,labels):
        observed.append((t.clone(),labels.clone()));return torch.zeros_like(x)
    heat=tccm_heatmap(zero,data,labels=torch.tensor([0,1]))
    torch.testing.assert_close(heat[:,0,0],torch.tensor([5.,2.]))
    assert torch.equal(observed[0][0],torch.ones(2))
    assert torch.equal(observed[0][1],torch.tensor([0,1]))


def test_wt_reverse_flow_target_and_energy_are_directionally_correct():
    data=torch.tensor([[[[1.,3.]]]])
    mapped=wt_affine(data,mean=torch.tensor([[[[2.]]]]),variance=torch.tensor([[[[1.]]]]),epsilon=1e-6)
    torch.testing.assert_close(mapped,torch.tensor([[[[-1.,1.]]]]),atol=1e-6,rtol=1e-6)
    noise=torch.ones_like(mapped)*3
    target=noise-mapped
    assert wt_flow_objective(lambda x,t:target,mapped,noise,torch.tensor([.3])).item()==0
    seen=[]
    def integrate(x,a,b,labels):
        seen.append((a,b));return torch.ones_like(x)*2
    torch.testing.assert_close(wt_flow_heatmap(mapped,integrate),torch.ones(1,1,2)*2)
    assert seen==[(0.,1.)]


def test_reconflow_requires_ten_paths_half_time_and_pathwise_min_l2():
    data=torch.ones(2,3,1,1)
    noise=torch.zeros((10,)+tuple(data.shape));noise[0]=data
    seen=[]
    def identity_integrator(x,a,b,labels):
        seen.append((a,b,labels.clone()));return x
    heat=reconflow_heatmap(data,noise,identity_integrator,labels=torch.tensor([4,7]))
    assert torch.equal(heat,torch.zeros(2,1,1))
    assert seen[0][:2]==(.5,1.)
    assert torch.equal(seen[0][2],torch.tensor([4,7]).repeat(10))
    with pytest.raises(ValueError,match='ten'):
        reconflow_heatmap(data,noise[:8],identity_integrator)


def test_invalid_masks_statistics_labels_and_endpoints_are_rejected():
    data=torch.ones(2,3,2,2)
    with pytest.raises(ValueError,match='binary'):
        reflect_corruption(data,torch.full_like(data,.5),alpha=torch.tensor(1.),beta=torch.tensor(0.),spatial_noise=data,scalar_noise=torch.tensor(0.))
    with pytest.raises(ValueError,match='per image'):
        tccm_heatmap(lambda x,t:x,data,labels=torch.tensor([1]))
    with pytest.raises(ValueError,match='positive epsilon'):
        wt_affine(data,mean=torch.zeros(2,1,1,1),variance=-torch.ones(2,1,1,1),epsilon=1e-6)
    with pytest.raises(ValueError,match='endpoint shape'):
        wt_flow_heatmap(data,lambda *args:torch.ones(1))


def recipe():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/'configs/experiments/fm_flow_mismatching_original_protocol.v1.json').read_text(encoding='utf-8'))


def test_full_native_protocol_retains_520_claims_without_measured_values():
    config=recipe()
    assert validate_protocol(config)==520
    assert not config['all_paper_tables_and_toy_experiments_transcribed']
    assert not config['full_original_raw_data_verified']


@pytest.mark.parametrize('change',['default8','short_training','old_sweep','missing_epoch','TCCM2000','certify_TAB'])
def test_protocol_rejects_development_budget_and_false_certification(change):
    config=copy.deepcopy(recipe())
    if change=='default8':config['original_score']['paths']=8
    elif change=='short_training':config['original_training']['epochs']=100
    elif change=='old_sweep':config['Table6_compute'][0]['T']=32
    elif change=='missing_epoch':config['Table9_epoch_checkpoints'].pop()
    elif change=='TCCM2000':config['training_exceptions']['TCCM']['epochs']=2000
    elif change=='certify_TAB':config['strict_TAB_equivalent']=True
    with pytest.raises(ValueError):validate_protocol(config)
