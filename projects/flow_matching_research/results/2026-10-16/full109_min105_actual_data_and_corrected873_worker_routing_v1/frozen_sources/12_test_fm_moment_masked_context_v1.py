"""Protocol tests only; full official weight checks have their own guarded job."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from iia_benchmark.models import fm_moment_masked_context_v1 as s
from scripts.flow_matching import run_moment_masked_context_job_v1 as worker


class MaskAwareModel:
    patch_len=8
    def __call__(self,x_enc,input_mask,mask):
        assert torch.equal(input_mask,mask)
        self.inputs=x_enc.clone();self.mask=mask.clone()
        observed=mask[:,None,:].bool()
        mean=torch.where(observed,x_enc,0).sum(dim=-1,keepdim=True)/observed.sum(dim=-1,keepdim=True)
        from types import SimpleNamespace
        return SimpleNamespace(reconstruction=mean.expand_as(x_enc))


@pytest.mark.parametrize('length',[8,100,105,107,234,239,511])
def test_real_points_retained_and_fill_neither_observed_nor_scored(length):
    rng=np.random.default_rng(31);values=rng.normal(size=(length,3)).astype(np.float32);before=values.copy()
    a=s.MaskedContextWindows(values);b=s.MaskedContextWindows(values,fill_value=10000.)
    model=MaskAwareModel();first=s.reconstruct_masked_context(model,a,1,'cpu')
    assert model.mask.shape==(1,512) and int(model.mask.sum())==length
    assert torch.all(model.mask[:,length:]==0) and first.shape==(1,length)
    second=s.reconstruct_masked_context(model,b,1,'cpu')
    assert np.array_equal(first,second) and np.array_equal(values,before)
    windows,starts=worker.covering_windows(values,512)
    assert isinstance(windows,s.MaskedContextWindows) and windows.valid_length==length
    assert np.array_equal(worker.align_scores(first,starts,length,512),first[0])


@pytest.mark.parametrize('length',[512,513,1024,1041])
def test_full_context_windows_and_alignment_exactly_preserve_original_worker(length):
    values=np.arange(length*2,dtype=np.float32).reshape(length,2)
    a,sa=worker.covering_windows(values,512);b,sb=worker.original.covering_windows(values,512)
    assert not isinstance(a,s.MaskedContextWindows) and np.array_equal(a,b) and np.array_equal(sa,sb)
    scores=a.mean(-1)
    assert np.array_equal(worker.align_scores(scores,sa,length,512),worker.original.align_scores(scores,sb,length,512))


@pytest.mark.parametrize('fault',['plain','missing_length','batch','length','nonfinite','batch_size'])
def test_incomplete_or_ambiguous_observation_metadata_rejected(fault):
    windows=s.MaskedContextWindows(np.ones((105,2),np.float32));batch=1
    if fault=='plain':windows=np.asarray(windows)
    elif fault=='missing_length':windows.valid_length=None
    elif fault=='batch':windows=np.repeat(windows,2,axis=0)
    elif fault=='length':windows.valid_length=512
    elif fault=='nonfinite':windows.flat[-1]=np.nan
    else:batch=2
    with pytest.raises(ValueError):s.reconstruct_masked_context(MaskAwareModel(),windows,batch,'cpu')


@pytest.mark.parametrize('fault',['padding_targets','missing_target','wrong_start','nonfinite'])
def test_alignment_cannot_accept_fill_or_missing_real_targets(fault):
    scores=np.ones((1,105));starts=np.array([0])
    if fault=='padding_targets':scores=np.ones((1,512))
    elif fault=='missing_target':scores=scores[:,:104]
    elif fault=='wrong_start':starts=np.array([1])
    else:scores[0,-1]=np.nan
    with pytest.raises(ValueError):worker.align_scores(scores,starts,105,512)


def test_fit_concatenation_does_not_estimate_any_parameters(monkeypatch):
    from iia_benchmark.models import fm_moment_pretrained as original
    class FakeLoaded:
        def to(self,device):return self
    monkeypatch.setattr(original,'load_pretrained',lambda *a:(FakeLoaded(),{}))
    short=s.MaskedContextWindows(np.ones((107,2),np.float32));full=np.ones((1,512,2),np.float32)
    training=np.concatenate([short,full])
    model=s.MomentMaskedContextWindow('source','checkpoint','weights_sha','config_sha')
    model.fit(training)
    assert model.checkpoint_load_audit_['fit_points_used_for_parameter_estimation']==0
    assert model.checkpoint_load_audit_['fine_tuning_epochs']==0 and model.training_losses_==[]


@pytest.mark.parametrize('override',[dict(window=100),dict(batch_size=2),dict(epochs=1),dict(padding_policy='TAB_always_extra_patch'),dict(observation_protocol='all_observed')])
def test_original_context_zero_shot_budget_and_mask_policy_frozen(override):
    with pytest.raises(ValueError):s.MomentMaskedContextWindow('source','checkpoint','weights_sha','config_sha',**override)
