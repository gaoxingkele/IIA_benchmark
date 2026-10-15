"""Independent analytic scoring and corrupted-checkpoint numerical unit tests."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_legacy_window_adapter import LegacyWindowAdapter
from scripts.flow_matching import legacy_baseline_checkpoint_math_v1 as m


def test_uniform_attention_has_exact_uniform_energy_at_all_temperatures():
    s=[np.full((2,3,4,4),.25,dtype=np.float32)]*2
    for temperature in [1.,50.]:
        values,count=m.association_numpy(s,s,temperature)
        assert np.array_equal(values,np.full((2,4),.25))
        assert count==2*2*2*3*4*4


def test_all_heads_are_reduced_after_per_head_KL_not_before():
    s=np.array([[[[.9,.1],[.8,.2]],[[.1,.9],[.2,.8]]]])
    p=np.array([[[[.5,.5],[.5,.5]],[[.5,.5],[.5,.5]]]])
    values,_=m.association_numpy([s],[p],50.)
    energy=[]
    for query in range(2):
        total=0.
        for head in range(2):
            for key in range(2):
                a=float(s[0,head,query,key]);b=.5
                total+=(a-b)*(np.log(a+1e-4)-np.log(b+1e-4))/2
        energy.append(-50*total)
    wanted=np.exp(np.array(energy)-max(energy));wanted/=wanted.sum()
    assert np.allclose(values[0],wanted,atol=1e-14,rtol=1e-14)
    assert not np.allclose(values,[.5,.5])


def test_source_AT_sums_layers_while_source_DC_averages_layers():
    s=np.array([[[[.9,.1],[.8,.2]]]])
    p=np.full_like(s,.5)
    summed,_=m.association_numpy([s,s,s],[p,p,p],1.,average_layers=False)
    single,_=m.association_numpy([s],[p],3.)
    averaged,_=m.association_numpy([s,s,s],[p,p,p],1.)
    assert np.allclose(summed,single,rtol=1e-14,atol=1e-14)
    assert not np.allclose(summed,averaged)


@pytest.mark.parametrize('fault',['empty','zero_prior','negative_series','nonfinite','shape'])
def test_invalid_or_incomplete_full_attention_rejected(fault):
    s=np.full((1,2,3,3),1/3);p=s.copy()
    if fault=='empty':
        with pytest.raises(ValueError):m.association_numpy([],[],50.)
        return
    if fault=='zero_prior':p[:]=0
    elif fault=='negative_series':s[0,0,0,0]=-.1
    elif fault=='nonfinite':p[0,0,0,0]=np.nan
    else:p=p[:,:,:,:2]
    with pytest.raises(ValueError):m.association_numpy([s],[p],50.)


@pytest.fixture(scope='module')
def trained():
    torch.set_num_threads(2)
    config=dict(parameters=dict(method='usad',parameters=dict(window=100,hidden_size=100,latent_size=40,
        batch_size=256,adversarial=False,alpha=.5,beta=.5),epochs=20,seed=1103,device='cpu'))
    source=LegacyWindowAdapter(**config['parameters'])
    data=np.random.default_rng(1103).normal(size=(3,100,2)).astype(np.float32);source.fit(data)
    checkpoint=dict(configuration=config,overrides=dict(seed=1103,device='cpu'),window=100,adjacency=None,
        modules={'_model':source._model.state_dict()})
    job=dict(window=100,seed=1103,parameter_overrides=checkpoint['overrides'])
    return source,data,checkpoint,job,config


def test_all_three_USAD_components_restore_and_both_full_errors_checked(trained):
    source,data,checkpoint,job,config=trained
    restored=m.restore_model(checkpoint,job,config,2);score,report=m.checked_score(restored)
    assert all(p.requires_grad for p in restored._model.parameters())
    assert np.array_equal(score(data),source.score(data))
    assert report['forward_calls']==3 and report['windows']==3
    assert report['reconstruction_coordinates_checked']==1200
    assert len(restored._model)==3


@pytest.mark.parametrize('fault',['module','missing_decoder','shape','dtype','nonfinite','configuration','overrides','adjacency'])
def test_corrupt_complete_trained_checkpoint_rejected(trained,fault):
    source,data,original,job,config=trained;checkpoint=deepcopy(original)
    state=checkpoint['modules']['_model'];key=next(iter(state))
    if fault=='module':checkpoint['modules']['unexpected']={}
    elif fault=='missing_decoder':
        for name in list(state):
            if name.startswith('2.'):state.pop(name)
    elif fault=='shape':state[key]=state[key][:-1]
    elif fault=='dtype':state[key]=state[key].double()
    elif fault=='nonfinite':state[key].reshape(-1)[0]=float('nan')
    elif fault=='configuration':checkpoint['configuration']={}
    elif fault=='overrides':checkpoint['overrides']=dict(seed=42,device='cpu')
    else:checkpoint['adjacency']=torch.eye(2)
    with pytest.raises(ValueError):m.restore_model(checkpoint,job,config,2)


def test_incorrect_native_weighted_error_is_rejected_even_with_correct_forward_outputs(trained):
    source,data,checkpoint,job,config=trained;restored=m.restore_model(checkpoint,job,config,2)
    original=restored.score;restored.score=lambda values:original(values)+.01
    checked,report=m.checked_score(restored)
    with pytest.raises(ValueError):checked(data)
