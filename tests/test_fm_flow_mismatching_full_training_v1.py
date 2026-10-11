"""Contract, tail coverage and resumed stochastic optimizer state, no benchmark."""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import (
    ROOT, validate_history, validate_job, verify_receipts,
)
from scripts.flow_matching.run_flow_mismatching_full_training_v1 import (
    capture_rng, restore_rng, save_checkpoint, train_epoch,
)
from iia_benchmark.models.fm_supplemental import image_cfm_objective
from iia_benchmark.models.fm_optimizer_reference import build_verified_dinomaly_stableadamw


def contract_fixture():
    model = json.loads((ROOT / 'configs/models/fm_flow_mismatching_mvtec_original.v1.json').read_text(encoding='utf-8'))
    optimizer = json.loads((ROOT / 'configs/models/fm_flow_mismatching_stableadamw_dinomaly_reference.v1.json').read_text(encoding='utf-8'))
    dataset = dict(expected_original_category_count=15, expected_native_splits=dict(train=3629),
                   complete_raw_data_verified=True, complete_derived_data_verified=True)
    job = dict(variant='conditional', seed=1103, expected_training_images_per_epoch=3629,
        expected_optimizer_steps=2000 * 114,
        training=dict(epochs=2000, batch_size=32, gradient_accumulation=1,
            mixed_precision='bfloat16', TF32=True, early_stopping=False, scheduler=None,
            EMA=False, drop_last=False, shuffle=True, distributed_data_parallel=False,
            world_size=1, cpu_threads=2, checkpoint_interval=100))
    return job, model, optimizer, dataset


@pytest.mark.parametrize('field,value', [('epochs', 1999), ('batch_size', 16),
    ('gradient_accumulation', 2), ('early_stopping', True), ('mixed_precision', 'float32'),
    ('drop_last', True)])
def test_refuse_reduced_or_changed_original_budget(field, value):
    job, model, optimizer, dataset = contract_fixture()
    assert validate_job(job, model, optimizer, dataset) == 3629
    job['training'][field] = value
    with pytest.raises(ValueError):
        validate_job(job, model, optimizer, dataset)


def test_conditioning_ablation_retains_full_architecture_and_dataset():
    job, model, optimizer, dataset = contract_fixture()
    job['variant'] = 'pooled'
    with pytest.raises(ValueError):
        validate_job(job, model, optimizer, dataset)
    model['parameters']['num_classes'] = None
    assert validate_job(job, model, optimizer, dataset) == 3629
    model['parameters']['model_channels'] = 32
    with pytest.raises(ValueError):
        validate_job(job, model, optimizer, dataset)


def test_subset_optimizer_and_missing_audit_rejected():
    original = contract_fixture()
    for kind in ('subset', 'optimizer', 'audit', 'seed'):
        job, model, optimizer, dataset = copy.deepcopy(original)
        if kind == 'subset': job['expected_training_images_per_epoch'] = 100
        if kind == 'optimizer': optimizer['parameters']['amsgrad'] = False
        if kind == 'audit': dataset['complete_derived_data_verified'] = False
        if kind == 'seed': job['seed'] = 42
        with pytest.raises(ValueError):
            validate_job(job, model, optimizer, dataset)


class TinyFixture(torch.nn.Module):
    def __init__(self, conditional):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))
        self.dropout = torch.nn.Dropout(0.2)
        self.embedding = torch.nn.Embedding(3, 3) if conditional else None

    def forward(self, images, time, labels=None):
        value = torch.einsum('ij,bjhw->bihw', self.weight, self.dropout(images))
        if self.embedding is not None:
            value = value + self.embedding(labels)[:, :, None, None]
        return value + time[:, None, None, None]


def native_optimizer(model):
    _, _, config, _ = contract_fixture()
    return build_verified_dinomaly_stableadamw(model.parameters(), **config['parameters'], asset_root=ROOT)


@pytest.mark.parametrize('conditional', [False, True])
def test_full_tail_stochastic_resume_is_bitwise_equal_on_cpu_fixture(tmp_path, conditional):
    torch.set_num_threads(2)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    images = np.arange(5 * 3 * 4 * 4, dtype=np.float32).reshape(5, 3, 4, 4) / 100
    categories = np.arange(5, dtype=np.int64) % 3
    model = TinyFixture(conditional); optimizer = native_optimizer(model)
    order = torch.Generator().manual_seed(9)
    kwargs = dict(conditional=conditional, batch_size=2, device=torch.device('cpu'),
                  bf16=False, torch=torch, np=np, objective=image_cfm_objective)
    first = train_epoch(model, optimizer, images, categories, order=order, **kwargs)
    assert first['images_visited'] == first['unique_images_visited'] == 5
    assert first['optimizer_steps'] == 3  # full final batch of one, not two steps
    checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                      rng=capture_rng(torch, np, order))
    path = tmp_path / 'resume.pt'
    save_checkpoint(path, checkpoint, torch)
    saved_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = train_epoch(model, optimizer, images, categories, order=order, **kwargs)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restored = TinyFixture(conditional); restored_optimizer = native_optimizer(restored)
    old = torch.load(path, weights_only=False)
    restored.load_state_dict(old['model']); restored_optimizer.load_state_dict(old['optimizer'])
    restored_order = torch.Generator(); restore_rng(old['rng'], torch, np, restored_order)
    actual = train_epoch(restored, restored_optimizer, images, categories, order=restored_order, **kwargs)
    assert actual == expected
    for key in expected_weights:
        assert torch.equal(expected_weights[key], restored.state_dict()[key])
    for key, values in expected_optimizer['state'].items():
        for name, value in values.items():
            got = restored_optimizer.state_dict()['state'][key][name]
            assert torch.equal(value, got) if torch.is_tensor(value) else value == got
    save_checkpoint(path, dict(fixture_second_state=True), torch)
    assert hashlib.sha256((tmp_path / 'resume_previous.pt').read_bytes()).hexdigest() == saved_hash
    # Synthetic fixtures prove plumbing only, never original image accuracy or CUDA equivalence.


def test_missing_epoch_or_tail_coverage_rejected():
    history = [dict(epoch=1, images_visited=3629, unique_images_visited=3629,
        optimizer_steps=114, mean_training_loss=1.0, permutation_sha256='a' * 64)]
    validate_history(history, 1, 3629)
    with pytest.raises(ValueError): validate_history(history, 2, 3629)
    history[0]['unique_images_visited'] = 3628
    with pytest.raises(ValueError): validate_history(history, 1, 3629)


def test_source_corruption_rejected(tmp_path):
    path = tmp_path / 'source.py'; path.write_bytes(b'original')
    receipt = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    verify_receipts([receipt])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): verify_receipts([receipt])
