from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.flow_matching import recover_moment_CICIDS_full_durable_v1 as s


def test_original_78_contract_preserved_and_only_commit_admission_strengthened():
    c=s.registration();p=s.read(s.ROOT/s.PARENT)
    assert c['settings']==dict(p['settings'],minimum_commit_headroom_bytes=40*2**30)
    assert c['resource_lock']==p['resource_lock']
    assert c['original_job'] in p['required_ids'] and len(p['required_ids'])==78
    assert c['native_CPU_environment_and_batch_and_all_windows_unchanged']
    assert c['artifact_root']!=p['artifact_root'] and c['state_root']!=p['state_root']
    assert not c['all_original_experiments_complete']


def inputs(value=1.):
    return dict(x_enc=torch.full((1,3,8),value),input_mask=torch.ones(1,8),mask=torch.ones(1,8))


def test_complete_forward_checkpoint_resume_is_exact_and_skips_only_verified_forward(tmp_path):
    calls=[]
    def source(**kw):
        calls.append(1);return SimpleNamespace(reconstruction=kw['x_enc']*.7+.2)
    a=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():first=a.wrap(source)(**inputs())
    assert a.actual_new_forwards==1 and a.calls==1
    b=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():second=b.wrap(source)(**inputs())
    assert torch.equal(first.reconstruction,second.reconstruction)
    assert len(calls)==1 and b.resumed_forwards==1 and b.actual_new_forwards==0


def test_full_input_and_mask_values_and_ordinal_are_part_of_cache_identity(tmp_path):
    calls=[]
    def source(**kw):
        calls.append(1);return SimpleNamespace(reconstruction=kw['x_enc'].clone())
    a=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():
        a.wrap(source)(**inputs());a.wrap(source)(**inputs())
    b=s.FullForwardCache(tmp_path,{'source':'fixed'})
    changed=inputs();changed['mask'][0,-1]=0
    with torch.inference_mode():b.wrap(source)(**changed)
    c=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():c.wrap(source)(**inputs(2.))
    d=s.FullForwardCache(tmp_path,{'source':'different_checkpoint'})
    with torch.inference_mode():d.wrap(source)(**inputs())
    assert len(calls)==5 and a.actual_new_forwards==2 and b.actual_new_forwards==1 and c.actual_new_forwards==1
    assert d.actual_new_forwards==1


def test_cache_tampering_and_incomplete_forward_are_never_accepted(tmp_path):
    a=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():a.wrap(lambda **kw:SimpleNamespace(reconstruction=kw['x_enc'].clone()))(**inputs())
    array=next(tmp_path.glob('*/reconstruction.npy'))
    with array.open('ab') as f:f.write(b'changed')
    b=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode(),pytest.raises(ValueError):b.wrap(lambda **kw:None)(**inputs())
    c=s.FullForwardCache(tmp_path/'other',{'source':'fixed'})
    with torch.inference_mode(),pytest.raises(ValueError):
        c.wrap(lambda **kw:SimpleNamespace(reconstruction=kw['x_enc'][...,:3]))(**inputs())
    assert not list((tmp_path/'other').glob('*/receipt.json'))


def test_interrupted_pending_directory_is_preserved_and_ignored(tmp_path):
    orphan=tmp_path/'pending-old';orphan.mkdir();(orphan/'reconstruction.npy').write_bytes(b'incomplete')
    a=s.FullForwardCache(tmp_path,{'source':'fixed'})
    with torch.inference_mode():a.wrap(lambda **kw:SimpleNamespace(reconstruction=kw['x_enc'].clone()))(**inputs())
    assert (orphan/'reconstruction.npy').read_bytes()==b'incomplete'
    assert a.actual_new_forwards==1 and a.resumed_forwards==0


def test_original_full_window_MSE_native_alignment_and_tails_match_cold_and_warm_cache(tmp_path):
    # Numerical plumbing fixture only; no pretrained MOMENT or benchmark claim.
    from iia_benchmark.models.fm_moment_pretrained import reconstruct_windows
    from scripts.flow_matching.run_moment_pretrained_job import segment_scores
    calls=[]
    class Fixture(torch.nn.Module):
        patch_len=4
        def forward(self,**kw):
            calls.append(1)
            return SimpleNamespace(reconstruction=kw['x_enc']*.7+.2)
    groups=[np.arange(n*3,dtype=np.float32).reshape(n,3)/23 for n in [19,3,21]]
    def execute(cache=None):
        holder=SimpleNamespace(model=Fixture().eval())
        holder.score=lambda windows:reconstruct_windows(holder.model,windows,1,'cpu','minimal_patch')
        if cache:holder.model.forward=cache.wrap(holder.model.forward)
        checked,record=s.original.observed_score(holder,8,dict(rtol=2e-6,atol=2e-6))
        holder.score=checked
        values=[segment_scores(holder,v,8) for v in groups]
        return values,record
    baseline,baseline_record=execute()
    cold=s.FullForwardCache(tmp_path,{'actual_fixture_protocol':'original_CPU_batch1'})
    cold_values,cold_record=execute(cold)
    warm=s.FullForwardCache(tmp_path,{'actual_fixture_protocol':'original_CPU_batch1'})
    warm_values,warm_record=execute(warm)
    assert baseline_record==cold_record==warm_record
    assert cold.actual_new_forwards==warm.resumed_forwards==7 and warm.actual_new_forwards==0
    assert len(calls)==14
    for before,(cold_score,cold_audit),(warm_score,warm_audit) in zip(baseline,cold_values,warm_values):
        score,audit=before
        assert score.dtype==cold_score.dtype==warm_score.dtype
        assert score.tobytes()==cold_score.tobytes()==warm_score.tobytes()
        assert audit==cold_audit==warm_audit and audit['native_points']==len(score)
