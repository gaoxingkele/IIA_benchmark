"""Native source and analytical CPU component checks; never benchmark results."""
import random
import types
import numpy as np
import pytest
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read
from iia_benchmark.models.fm_reflect_wt_native import (
    load_verified_reflect_masks,reflect_native_mask,load_verified_wt_flow,
    wt_native_standardize,wt_native_objective,wt_native_endpoint)


def config():return read(ROOT/'configs/models/fm_reflect_wt_native_components.v1.json')


def core(kind):
    cfg=config()[kind]
    loader=load_verified_reflect_masks if kind=='reflect' else load_verified_wt_flow
    return loader(cfg['source_path'],cfg['expected_sha256'],asset_root=ROOT)


@pytest.fixture(autouse=True)
def restore_cpu_and_native_random_state():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
    yield
    random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
    torch.set_num_threads(threads)


@pytest.mark.parametrize('kind',['reflect','wt'])
def test_original_source_change_is_rejected_without_overwrite(tmp_path,kind):
    cfg=config()[kind];original=(ROOT/cfg['source_path']).read_bytes()
    native=core(kind);assert native.source_sha256==cfg['expected_sha256']
    changed=tmp_path/'changed.py';changed.write_bytes(original+b'\n# modified original\n')
    loader=load_verified_reflect_masks if kind=='reflect' else load_verified_wt_flow
    with pytest.raises(ValueError,match='SHA256'):loader(changed,cfg['expected_sha256'])
    assert (ROOT/cfg['source_path']).read_bytes()==original


@pytest.mark.parametrize('source',['BraTS2021Dataset','ATLASDataset'])
def test_native_mask_exact_rng_replay_matches_direct_author_method(source):
    native=core('reflect');roi=np.ones((32,32),dtype=np.uint8)
    roi[:4,:]=0;roi[:,:4]=0
    random.seed(181);np.random.seed(239)
    python=random.getstate();numpy=np.random.get_state()
    direct=getattr(native,source)(types.SimpleNamespace(embedding_size=32,embedding_dim=3),roi,4)
    next_python=random.random();next_numpy=np.random.random()
    random.setstate(python);np.random.set_state(numpy)
    actual=reflect_native_mask(native,roi,channels=3,max_objects=4,source_dataset=source)
    assert np.array_equal(actual,direct)
    assert random.random()==next_python and np.random.random()==next_numpy
    assert actual.shape==(3,32,32) and actual.dtype==np.uint8
    assert np.array_equal(actual[0],actual[1]) and np.array_equal(actual[1],actual[2])
    assert set(np.unique(actual))<=set(range(5))
    assert not actual[:,:4,:].any() and not actual[:,:,:4].any()


def test_mask_requires_explicit_nonempty_roi_and_geometry():
    native=core('reflect')
    for roi in (np.zeros((32,32),dtype=np.uint8),np.ones((32,16),dtype=np.uint8),np.full((32,32),.5)):
        with pytest.raises(ValueError,match='ROI'):reflect_native_mask(native,roi,channels=3,max_objects=4,source_dataset='ATLASDataset')
    with pytest.raises(ValueError):reflect_native_mask(native,np.ones((32,32)),channels=3,max_objects=4,source_dataset='MVTec')


def test_full128rgb_native_standardization_matches_fresh_author_layer_norm():
    epsilon=config()['wt']['related_native_LayerNorm_epsilon']
    images=torch.randn(2,3,128,128,dtype=torch.float64)*3.+7.
    actual=wt_native_standardize(images,epsilon=epsilon)
    expected=torch.nn.LayerNorm(images.shape[1:],eps=epsilon,dtype=torch.float64)(images)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    manual=(images-images.mean((1,2,3),keepdim=True))/torch.sqrt(images.var((1,2,3),unbiased=False,keepdim=True)+epsilon)
    torch.testing.assert_close(actual,manual,rtol=1e-12,atol=1e-12)
    assert config()['wt']['epsilon_required_argument']


def test_native_reverse_fm_objective_and_gradient_match_independent_equation():
    native=core('wt');images=torch.randn(2,3,4,5,dtype=torch.float64)
    noise=torch.randn_like(images);time=torch.tensor([.2,.7],dtype=torch.float64)
    parameter=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([2,4]);calls=[]
    def velocity(state,t,y):calls.append((state.detach().clone(),t.clone(),y.clone()));return parameter*state
    loss=wt_native_objective(velocity,images,time,noise,native_core=native,epsilon=1e-5,labels=labels)
    mean=images.mean((1,2,3),keepdim=True)
    mapped=(images-mean)/torch.sqrt((images-mean).square().mean((1,2,3),keepdim=True)+1e-5)
    state=(1-time[:,None,None,None])*mapped+time[:,None,None,None]*noise
    residual=parameter.detach()*state-(noise-mapped)
    torch.testing.assert_close(loss,residual.square().mean(),rtol=1e-12,atol=1e-12)
    loss.backward();torch.testing.assert_close(parameter.grad,2*(residual*state).mean(),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(calls[0][0],state,rtol=1e-12,atol=1e-12)
    assert torch.equal(calls[0][1],time) and torch.equal(calls[0][2],labels)


def test_native_left_grid_euler_full128rgb_matches_closed_form_affine_ode_discretization():
    native=core('wt');image=torch.randn(2,3,128,128,dtype=torch.float64);times=[]
    def velocity(state,time):times.append(time);return .25*state+.1
    actual=wt_native_endpoint(velocity,image,native_core=native,steps=10)
    factor=(1+.25/10)**10
    expected=factor*image+(.1/.25)*(factor-1)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert [float(time[0]) for time in times]==[j/10 for j in range(10)]
    assert all(time.shape==(2,) for time in times)
    assert image.shape==actual.shape and not actual.requires_grad
    for invalid in (0,-1,.5,True):
        with pytest.raises(ValueError):wt_native_endpoint(velocity,image,native_core=native,steps=invalid)


def test_original_feature_and_target_image_energy_oracle_selection_are_distinguished():
    cfg=config()
    assert 'mean' in cfg['wt']['native_feature_energy']
    assert 'SUM' in cfg['wt']['target_image_energy']
    assert cfg['wt']['strict_TAB_compatible_native_checkpoint_selection'] is False
    assert cfg['original_author_equivalence_certified'] is False
    assert cfg['benchmark_performance'] is False
