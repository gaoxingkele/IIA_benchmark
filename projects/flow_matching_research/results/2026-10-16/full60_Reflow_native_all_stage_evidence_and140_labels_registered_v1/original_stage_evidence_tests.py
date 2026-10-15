"""Perturb persisted evidence; these numerical unit cases are not performance."""
from copy import deepcopy
import json
import pytest
import numpy as np
import torch
from iia_benchmark.models.fm_reflow import ReflowCFMDetector,transport_endpoints
from scripts.flow_matching import reflow_checkpoint_evidence_v1 as e


@pytest.mark.parametrize('solver',['euler','heun'])
def test_every_ODE_step_input_and_discrete_action_match_analytic_constant_velocity(solver):
    origins=torch.tensor([[0.,1.],[2.,-1.]])
    def teacher(x):return torch.full_like(x[:,:-1],.5)
    observed=e.CheckedTeacher(teacher,origins,50,solver)
    terminal,record=transport_endpoints(observed,origins,50,solver)
    counts=observed.finish(terminal,record)
    assert torch.allclose(terminal,origins+.5,atol=1e-6,rtol=1e-6)
    assert counts['forward_calls']==50*(2 if solver=='heun' else 1)
    assert counts['checked_state_coordinates']==origins.numel()*counts['forward_calls']


def test_wrong_ODE_state_or_missing_corrector_rejected():
    origins=torch.zeros(2,2)
    observed=e.CheckedTeacher(lambda x:x[:,:-1],origins,50,'heun')
    with pytest.raises(ValueError):observed(torch.ones(2,3))
    observed(torch.zeros(2,3))
    with pytest.raises(ValueError):observed.finish(origins,{})


def test_byte_equality_distinguishes_signed_zero_and_dtype():
    with pytest.raises(ValueError):e.tensor_bytes(torch.tensor([-0.]),torch.tensor([0.]))
    with pytest.raises(ValueError):e.tensor_bytes(torch.tensor([0.]),torch.tensor([0.],dtype=torch.float64))


@pytest.fixture(scope='module')
def trained():
    torch.set_num_threads(2)
    # Full original width,window,batch,epochs and50Heun retained;only input tiny.
    model=ReflowCFMDetector(objective='rectified',sigma=0.,window=100,hidden_size=128,
        epochs=20,reflow_stages=3,reflow_epochs=20,reflow_ode_steps=50,reflow_solver='heun',
        batch_size=64,seed=1103,device='cpu')
    model.fit(np.random.default_rng(1103).normal(size=(6,100,2)).astype(np.float32))
    return model,dict(total_training_epochs=60,additional_pair_generation_compute=True),dict(training_losses=model.training_losses_)


def test_every_stage_teacher_and_all_origins_endpoints_regenerate_exactly(trained):
    model,job,result=trained
    proof=e.verify_all_couplings(model,job,result,6)
    assert len(proof['stages'])==2
    for stage in proof['stages']:
        assert stage['pairs']==6 and stage['teacher_forward_calls']==100
        assert stage['independent_state_program_checked_coordinates']==120000
        assert stage['all_source_teacher_geometry_exact']
    assert not proof['optimizer_steps_or_warm_start_trajectory_independently_replayed']


@pytest.mark.parametrize('fault',['teacher','origin','terminal','missing_buffer','extra_buffer','metadata','pair_seed','pair_digest','geometry','warm_start','optimizer_reset','losses','pair_count','stage_count'])
def test_corrupt_actual_trained_stage_evidence_rejected(trained,fault):
    original,job,result=trained;model=deepcopy(original);result=deepcopy(result)
    evidence=model._reflow_evidence
    if fault in ['teacher','origin','terminal']:
        key={'teacher':'stage2_teacher0','origin':'stage2_z0','terminal':'stage2_z1'}[fault]
        with torch.no_grad():getattr(evidence,key).reshape(-1)[0]+=.125
    elif fault=='missing_buffer':delattr(evidence,'stage3_z1')
    elif fault=='extra_buffer':evidence.register_buffer('unexpected',torch.zeros(1))
    elif fault=='losses':result['training_losses']=result['training_losses'][:-1]
    elif fault=='pair_count':
        with pytest.raises(ValueError):e.verify_all_couplings(model,job,result,5)
        return
    else:
        records=json.loads(bytes(evidence.metadata_json_utf8.tolist()).decode())
        if fault=='metadata':raw=json.dumps(records,indent=2).encode()
        else:
            if fault=='pair_seed':records[1]['pair_seed']+=1
            elif fault=='pair_digest':records[1]['pair_sha256']='0'*64
            elif fault=='geometry':records[1]['teacher_discrete_excess_action_mean']+=1.
            elif fault=='warm_start':records[1]['warm_start']=False
            elif fault=='optimizer_reset':records[1]['optimizer_reset']=False
            elif fault=='stage_count':records.pop()
            raw=json.dumps(records,sort_keys=True,allow_nan=False).encode()
        evidence.metadata_json_utf8=torch.tensor(list(raw),dtype=torch.uint8)
    with pytest.raises((ValueError,KeyError)):e.verify_all_couplings(model,job,result,6)
