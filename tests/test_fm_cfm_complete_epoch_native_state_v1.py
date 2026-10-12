"""Full-architecture checkpoint diagnostics on tiny fixtures, never benchmarks."""
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_cfm_ts_original import CFMTSPaperTrajectory
from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import (
    CompleteEpochLane,bind_fit,bitwise_state,backend_policy)
from scripts.flow_matching.prepare_cfm_ts_original_data import sha


class FixtureInterruption(RuntimeError):
    pass


class InterruptedLane(CompleteEpochLane):
    def save_epoch(self,model,optimizer,epoch,prepared,node_data,count):
        super().save_epoch(model,optimizer,epoch,prepared,node_data,count)
        if epoch==2:
            raise FixtureInterruption('Controlled diagnostic interruption at a complete epoch')


def full_network():
    # The architecture and solver are the frozen production model. The short
    #fixture budget validates persistence only and receives no experiment credit.
    return CFMTSPaperTrajectory(method='node',target_mode='not_applicable',epochs=4,
        batch_size=258,grid_points=160,seed=117,input_projection=64,hidden=258,hidden_layers=3,
        solver='dopri5',rtol=1e-7,atol=1e-9,max_num_steps=10000)


def data():
    times=np.linspace(0,.12,4)
    return [(times,np.sin(times)[:,None])]


def job():
    return dict(id='diagnostic_fixture_only',seed=117,epochs=4,expected_optimizer_updates=4,
                dataset='synthetic_persistence_fixture_not_benchmark',batch_size=258)


@pytest.fixture(autouse=True)
def cpu_policy():
    before=backend_policy(torch)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    np.random.seed(117);random.seed(117);torch.manual_seed(117)
    yield
    torch.set_num_threads(before['cpu_threads'])
    torch.use_deterministic_algorithms(before['deterministic_algorithms'],warn_only=before['deterministic_warn_only'])


@pytest.fixture
def saved_boundary(tmp_path):
    model=full_network();lane=InterruptedLane(job(),tmp_path/'first',interval=2)
    model.fit=bind_fit(model,lane)
    with pytest.raises(FixtureInterruption):model.fit(data())
    path=tmp_path/'first/complete_epoch_checkpoints/epoch_0002/full_state.pt'
    assert path.is_file()
    return model,lane,path


def test_actual_full_network_checkpoint_save_does_not_change_original_training(tmp_path):
    baseline=full_network();baseline.fit(data())
    observed=full_network();lane=CompleteEpochLane(job(),tmp_path/'observed',interval=2)
    observed.fit=bind_fit(observed,lane);observed.fit(data())
    assert bitwise_state(baseline.model.state_dict(),observed.model.state_dict())
    assert baseline.loss_history_==observed.loss_history_
    assert baseline.model.nfe==observed.model.nfe and baseline.optimizer_updates_==observed.optimizer_updates_==4
    assert sum(p.numel() for p in observed.model.parameters())>150000
    assert np.array_equal(baseline.simulate([0.],np.linspace(0,.12,4)),observed.simulate([0.],np.linspace(0,.12,4)))
    assert not torch.cuda.is_initialized()


def test_actual_complete_restore_and_remaining_training_match_uninterrupted_weights_optimizer_losses_and_predictions(tmp_path,saved_boundary):
    _,_,path=saved_boundary
    baseline=full_network();whole=CompleteEpochLane(job(),tmp_path/'whole',interval=2)
    baseline.fit=bind_fit(baseline,whole);baseline.fit(data())
    resumed=full_network();continuation=CompleteEpochLane(job(),tmp_path/'continuation',interval=2,
        resume=path,resume_sha256=sha(path))
    resumed.fit=bind_fit(resumed,continuation);resumed.fit(data())
    assert bitwise_state(baseline.model.state_dict(),resumed.model.state_dict())
    assert bitwise_state(whole.last_optimizer_state,continuation.last_optimizer_state)
    assert baseline.loss_history_==resumed.loss_history_ and baseline.model.nfe==resumed.model.nfe
    assert np.array_equal(baseline.simulate([0.],np.linspace(0,.12,4)),resumed.simulate([0.],np.linspace(0,.12,4)))


@pytest.mark.parametrize('corruption',['bad_hash','partial_epoch','wrong_adam_step','changed_input','different_policy','nonfinite_weight'])
def test_actual_resume_rejects_partial_or_incompatible_native_state(tmp_path,saved_boundary,corruption):
    _,_,original=saved_boundary
    path=original;expected_sha=sha(original);state=torch.load(original,map_location='cpu',weights_only=False)
    if corruption=='bad_hash':expected_sha='0'*64
    if corruption=='partial_epoch':state['optimizer_updates']-=1
    if corruption=='wrong_adam_step':next(iter(state['optimizer_state']['state'].values()))['step']+=1
    if corruption=='different_policy':state['backend_policy']['cpu_threads']+=1
    if corruption=='nonfinite_weight':next(iter(state['model_state'].values())).fill_(float('inf'))
    if corruption not in ('bad_hash','changed_input'):
        path=tmp_path/'corrupted.pt';torch.save(state,path);expected_sha=sha(path)
    model=full_network();lane=CompleteEpochLane(job(),tmp_path/'rejected',interval=2,resume=path,resume_sha256=expected_sha)
    model.fit=bind_fit(model,lane)
    trajectories=data()
    if corruption=='changed_input':trajectories[0][1][0,0]+=1
    with pytest.raises(ValueError):model.fit(trajectories)


def test_actual_saved_epoch_is_preserved_on_duplicate_save(tmp_path,saved_boundary):
    _,_,path=saved_boundary
    before=sha(path)
    model=full_network();lane=CompleteEpochLane(job(),path.parents[2],interval=2)
    model.fit=bind_fit(model,lane)
    with pytest.raises(FileExistsError):model.fit(data())
    assert sha(path)==before
