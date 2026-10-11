import copy
import json

import pytest

from scripts.flow_matching import register_tsad_constant_fit_gpu_runtime_v2 as new
from scripts.flow_matching.run_light_controller import ROOT


@pytest.fixture
def registry():
    return json.loads((ROOT / new.REGISTRATION).read_text('utf-8'))


def test_every_actual_corrected_GPU_ablation_and_seed_retains_full_scientific_contract(registry):
    binding, queue = new.validate_scientific_controls(ROOT, registry)
    assert binding['jobs'] == len(queue['jobs']) == 170
    assert queue['evaluation']['cpu_threads'] == 4
    assert {j['parameter_overrides']['device'] for j in queue['jobs']} == {'cuda:0'}
    assert len({j['model_config'] for j in queue['jobs']}) == 34


def test_missing_affected_original_dataset_cannot_drop_controls(registry):
    registry['actual_affected_datasets'] = []
    with pytest.raises(ValueError, match='Every original affected'):
        new.validate_scientific_controls(ROOT, registry)


def test_wrong_corrected_manifest_cannot_pass_fidelity_gate(registry):
    registry['corrected_manifest'] = 'old_manifest.json'
    with pytest.raises(ValueError, match='Distinct corrected'):
        new.validate_scientific_controls(ROOT, registry)


def test_frozen_full_queue_cannot_be_replaced(registry):
    next(b for b in registry['queue_bindings'] if b['original_lane'] == 'gpu')['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Frozen all170'):
        new.validate_scientific_controls(ROOT, registry)
