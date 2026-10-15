"""Exact resource-failure selection and full-capacity synthetic checkpoint IO."""
from copy import deepcopy
import subprocess
import sys
import pytest
from scripts.flow_matching import run_cfm_sine_NODE1105_resource_recovery_v1 as s


def test_every285_slot_and_all_science_are_preserved_except_selected_execution_paths():
    c,q=s.registration();old=s.read(s.ROOT/s.ORIGINAL)
    assert len(q['jobs'])==285 and len(q['recovery_job_ids'])==1
    changed=[(a,b) for a,b in zip(old['jobs'],q['jobs']) if a!=b]
    assert len(changed)==1
    a,b=changed[0]
    assert a['id']==s.CANONICAL and s.science(a)==s.science(b)
    assert (b['epochs'],b['expected_optimizer_updates'],b['training_items'],b['batch_size'])==(10,20,320,256)
    assert (b['test_trajectories'],b['test_points_per_trajectory'],b['observed_dimensions'])==(80,500,1)
    assert c['checkpoint_interval']==1 and c['partial_epochs_credited']==0 and c['maximum_new_attempts']==1
    assert q['resource_policy']==dict(old['resource_policy'],minimum_commit_headroom_bytes=32*2**30)
    assert c['family_lock']=='experiments/runs/fm_cfm_ts_original_cpu.lock'
    assert not s.source_closed(c)


@pytest.mark.parametrize('fault',['epochs','budget','memory','emergency','lock','checkpoint','retry','stability'])
def test_reduced_budget_or_weakened_guards_or_repetition_rejected(monkeypatch,fault):
    c,q=s.registration();read=s.read
    monkeypatch.setattr(s,'read',lambda p:q if p==s.ROOT/s.QUEUE else read(p))
    if fault=='epochs':c['canonical_job']['epochs']=1
    elif fault=='budget':c['canonical_job']['expected_optimizer_updates']=2
    elif fault=='memory':c['resource_policy']['minimum_commit_headroom_bytes']=22*2**30
    elif fault=='emergency':c['resource_policy']['emergency_commit_headroom_bytes']=1
    elif fault=='lock':c['family_lock']='other.lock'
    elif fault=='checkpoint':c['checkpoint_interval']=10
    elif fault=='retry':c['maximum_new_attempts']=2
    else:c['stable_admission_samples']=1
    with pytest.raises(ValueError):s.verify(c)


def test_other284_replay_routes_and_entire_native_review_policy_are_preserved(monkeypatch):
    c,q=s.registration();sha=s.sha
    monkeypatch.setattr(s,'sha',lambda p:'0'*64 if p in [s.ROOT/s.CONFIG,s.ROOT/s.QUEUE] else sha(p))
    r=s.replay_registration(c,q);old=s.read(s.ROOT/s.PRIMARY)
    changed=[(a,b) for a,b in zip(old['routes'],r['routes']) if a!=b]
    assert len(changed)==1 and changed[0][0]['original_canonical_id']==s.CANONICAL
    assert s.science(changed[0][0]['job'])==s.science(changed[0][1]['job'])
    for key in ['python','cpu_threads','settings','resource_lock','environment_lock','prediction_tolerance','original_scope']:
        assert r[key]==old[key]
    assert r['selected_canonical_ids']==[s.CANONICAL]
    ns=s.replay_namespace()
    from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as original
    assert ns['review_worker'].__code__ is original.review_worker.__code__
    assert ns['verify_record'].__code__ is original.verify_record.__code__
    assert not ns['_selected_ready'](r['routes'][0],r)


def test_stable_admission_resets_after_low_memory_without_changing_thresholds(monkeypatch):
    c,_=s.registration();p=c['resource_policy']
    good=dict(physical_available_bytes=p['minimum_available_memory_bytes'],commit_available_bytes=p['minimum_commit_headroom_bytes'])
    samples=[good,good,dict(good,commit_available_bytes=good['commit_available_bytes']-1)]+[good]*6
    seen=[];sleeps=[];it=iter(samples)
    monkeypatch.setattr(s.time,'sleep',lambda n:sleeps.append(n))
    s.stable_headroom(c,dict(resource_snapshot=lambda:next(it)),seen.append)
    assert [r['stable_samples'] for r in seen]==[1,2,0,1,2,3,4,5,6]
    assert sleeps==[5]*8


def test_full_appendix_model_capacity10epochs_and_checkpoint_every_epoch_preserve_weights(tmp_path):
    import numpy as np
    import torch
    from iia_benchmark.models.fm_cfm_ts_original import CFMTSPaperTrajectory
    from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import CompleteEpochLane,bind_fit,bitwise_state,backend_policy
    c,_=s.registration();job=deepcopy(c['canonical_job'])
    parameters=dict(s.read(s.ROOT/job['model_config'])['parameters'],**job['model_overrides'])
    old=backend_policy(torch)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        times=np.linspace(0,.12,4);data=[(times,np.sin(times)[:,None]),(times,np.cos(times)[:,None])]
        baseline=CFMTSPaperTrajectory(**parameters);baseline.fit(data)
        observed=CFMTSPaperTrajectory(**parameters)
        fixture_job=dict(job,id='synthetic_checkpoint_probe_not_benchmark',training_items=2,expected_optimizer_updates=10)
        lane=CompleteEpochLane(fixture_job,tmp_path/'states',interval=1)
        observed.fit=bind_fit(observed,lane);observed.fit(data)
        assert baseline.optimizer_updates_==observed.optimizer_updates_==10
        assert baseline.loss_history_==observed.loss_history_ and len(observed.loss_history_)==10
        assert baseline.model.nfe==observed.model.nfe
        assert bitwise_state(baseline.model.state_dict(),observed.model.state_dict())
        assert sum(p.numel() for p in observed.model.parameters())>150000
        assert np.array_equal(baseline.simulate([0.],times),observed.simulate([0.],times))
        ledger=s.read(tmp_path/'states/completed_full_epoch_checkpoints.json')
        assert [r['completed_original_epochs'] for r in ledger]==list(range(1,11))
        assert all(r['additional_optimizer_updates']==0 and r['no_training_state_or_random_state_changed_by_save'] for r in ledger)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(old['cpu_threads'])
        torch.use_deterministic_algorithms(old['deterministic_algorithms'],warn_only=old['deterministic_warn_only'])


def test_registration_and_replay_bindings_are_cold_without_numpy_or_torch():
    code='from scripts.flow_matching import run_cfm_sine_NODE1105_resource_recovery_v1 as s;import sys;s.registration();s.replay_namespace();assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
