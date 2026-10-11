import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read
from scripts.flow_matching.full_image_training_execution_recovery_v2 import (
    native_path, write_json, save_checkpoint, enable_native_activation_recomputation)


def test_exact_260character_temporary_path_failure_is_repaired_without_shortening(tmp_path):
    base = tmp_path / 'long'
    tail = 260 - len(str(base)) - 1 - len('record.json') - 37
    path = base / ('a' * tail) / 'record.json'
    assert len(str(path)) + 37 == 260
    write_json(path, dict(original_budget=2000, batch_size=32))
    assert json.loads(native_path(path).read_text()) == dict(original_budget=2000, batch_size=32)
    assert not list(native_path(path.parent).glob('*.tmp'))


def test_extended_paths_and_full_checkpoint_rolling_hardlinks_preserve_immutable_snapshots(tmp_path):
    path = tmp_path / 'resume.pt'
    value = dict(model=dict(weight=torch.arange(100)), optimizer=dict(step=1), rng=torch.get_rng_state())
    save_checkpoint(path, value, torch)
    snapshot = tmp_path / 'epoch_0100.pt'; os.link(path, snapshot)
    original_bytes = snapshot.read_bytes()
    second = copy.deepcopy(value); second['optimizer']['step'] = 2
    save_checkpoint(path, second, torch)
    assert torch.load(path, weights_only=False)['optimizer']['step'] == 2
    assert torch.load(tmp_path/'resume_previous.pt', weights_only=False)['optimizer']['step'] == 1
    assert snapshot.read_bytes() == original_bytes


def model_config():
    from iia_benchmark.models.fm_supplemental_author import FlowMismatchUNet
    config=read(ROOT/'configs/models/fm_wt_mvtec_controlled_image_domain.v1.json')
    return FlowMismatchUNet,config


def test_all_full_backbone_parameters_and_native_blocks_preserved_on_meta_device():
    cls,config=model_config()
    with torch.device('meta'): model=cls(**config['parameters'],asset_root=ROOT)
    original=list(model.state_dict())
    counts=enable_native_activation_recomputation(model)
    assert counts==dict(ResBlock=54,AttentionBlock=29)
    assert sum(parameter.numel() for parameter in model.parameters())==45289539
    assert list(model.state_dict())==original


@pytest.mark.parametrize('bf16',[False,True])
def test_native_recomputation_preserves_dropout_RNG_loss_gradients_and_optimizer_steps(bf16):
    from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw
    cls,config=model_config()
    parameters=dict(config['parameters']);parameters.update(model_channels=32,num_res_blocks=1,
        channel_mult=[1,2],attention_resolutions=[1,2],num_classes=2)
    torch.manual_seed(731)
    original=cls(**parameters,asset_root=ROOT)
    # Nonzero final weights avoid a trivial all-zero residual-network gradient fixture.
    with torch.no_grad(): original.model.out[-1].weight.normal_(std=.01)
    copied=copy.deepcopy(original);enable_native_activation_recomputation(copied)
    images=torch.randn(2,3,16,16);time=torch.rand(2);labels=torch.tensor([0,1]);target=torch.randn_like(images)
    optimizer_config=read(ROOT/'configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json')
    optimizers=[build_verified_dinomaly_stableadamw(model.parameters(),**optimizer_config['parameters'],asset_root=ROOT)
                for model in (original,copied)]
    for step in range(3):
        rng=torch.get_rng_state()
        outcomes=[]
        for model,optimizer in zip((original,copied),optimizers):
            torch.set_rng_state(rng);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                loss=torch.nn.functional.mse_loss(model(images,time,labels),target)
            loss.backward()
            gradients=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            optimizer.step();outcomes.append((loss.detach(),gradients,torch.get_rng_state()))
        assert torch.equal(outcomes[0][2],outcomes[1][2])
        assert torch.equal(outcomes[0][0],outcomes[1][0])
        for left,right in zip(outcomes[0][1],outcomes[1][1]):
            if left is None: assert right is None
            else: assert torch.allclose(left,right,rtol=1e-6,atol=1e-7)
        for left,right in zip(original.parameters(),copied.parameters()):
            assert torch.allclose(left,right,rtol=1e-6,atol=1e-7)


def test_non_native_model_cannot_silently_claim_activation_recomputation():
    with pytest.raises(ValueError):enable_native_activation_recomputation(torch.nn.Linear(3,3))
