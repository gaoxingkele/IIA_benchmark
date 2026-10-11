from pathlib import Path
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_optimizer_reference import load_verified_dinomaly_stableadamw,build_verified_dinomaly_stableadamw

ROOT=Path(__file__).resolve().parents[1]


def native_class():
    config=json.loads((ROOT/'configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json').read_text(encoding='utf-8'))
    p=config['parameters']
    path=ROOT/p['source_path']
    if not path.is_file():pytest.skip('Immutable registered external original optimizer not installed')
    return load_verified_dinomaly_stableadamw(path,p['expected_sha256'])


@pytest.mark.parametrize('amsgrad',[False,True])
def test_native_updates_match_independent_numpy_moments_with_spike_and_zero(amsgrad):
    original=native_class();parameter=torch.nn.Parameter(torch.tensor([.3,-.5,.7],dtype=torch.float64))
    lr=.01;wd=.13;b1=.9;b2=.99;eps=1e-6
    optimizer=original([parameter],lr=lr,betas=(b1,b2),eps=eps,weight_decay=wd,amsgrad=amsgrad,clip_threshold=1.)
    expected=parameter.detach().numpy().copy();first=np.zeros(3);second=np.zeros(3);largest=np.zeros(3)
    gradients=[np.array(g) for g in [[0.,0.,0.],[.0001,.0002,-.0003],[10.,-20.,30.],[.01,-.03,.02],[0.,0.,0.]]]
    before_source=hashlib.sha256(Path(original.__iia_original_source_path__).read_bytes()).hexdigest()
    for step,gradient in enumerate(gradients,1):
        parameter.grad=torch.tensor(gradient,dtype=torch.float64);optimizer.step()
        first=b1*first+(1-b1)*gradient;second=b2*second+(1-b2)*gradient**2
        largest=np.maximum(largest,second)
        denominator=np.sqrt((largest if amsgrad else second)/(1-b2**step))+eps
        factor=max(1.,np.sqrt(np.mean((gradient/denominator)**2)))
        expected=expected*(1-lr*wd)-lr*first/((1-b1**step)*factor*denominator)
        np.testing.assert_allclose(parameter.detach().numpy(),expected,rtol=1e-12,atol=1e-12)
    assert hashlib.sha256(Path(original.__iia_original_source_path__).read_bytes()).hexdigest()==before_source


def test_native_state_resume_retains_amsgrad_and_skipped_parameter():
    original=native_class()
    first=torch.nn.Parameter(torch.tensor([.3,-.5],dtype=torch.float64))
    untouched=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    optimizer=original([first,untouched],amsgrad=True)
    first.grad=torch.tensor([.4,-.7],dtype=torch.float64);optimizer.step()
    assert torch.equal(untouched.detach(),torch.tensor([1.,2.],dtype=torch.float64)) and untouched not in optimizer.state
    resumed=torch.nn.Parameter(first.detach().clone());resumed_untouched=torch.nn.Parameter(untouched.detach().clone())
    continued=original([resumed,resumed_untouched],amsgrad=True);continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for gradient in [torch.tensor([8.,-4.],dtype=torch.float64),torch.zeros(2,dtype=torch.float64)]:
        first.grad=gradient.clone();resumed.grad=gradient.clone();optimizer.step();continued.step()
        assert torch.equal(first.detach(),resumed.detach())
        assert torch.equal(optimizer.state[first]['max_exp_avg_sq'],continued.state[resumed]['max_exp_avg_sq'])


def test_source_tampering_rejected_before_python_import(tmp_path):
    path=tmp_path/'StableAdamW.py';path.write_text('raise RuntimeError("Must never execute")',encoding='utf-8')
    with pytest.raises(ValueError,match='SHA256 differs'):load_verified_dinomaly_stableadamw(path,'0'*64)


def test_registered_positive_numeric_guards_are_required():
    with pytest.raises(ValueError,match='Positive registered'):
        build_verified_dinomaly_stableadamw([],source_path='not-accessed',expected_sha256='',
            learning_rate=1e-4,betas=(.9,.99),epsilon=1e-6,weight_decay=1e-4,amsgrad=True,update_rms_clip_threshold=0)
