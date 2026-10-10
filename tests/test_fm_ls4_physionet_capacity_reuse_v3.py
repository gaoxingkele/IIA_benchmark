import copy
import json
from pathlib import Path
import pytest
from scripts.flow_matching.verify_ls4_physionet_capacity_reuse_v3 import verify_capacity_reuse


def test_recovery_keeps_every_original_full_patient_model_seed_and_capacity():
    q = json.loads(Path('configs/experiments/fm_ls4_native_physionet_queue.v3.json').read_text())
    assert verify_capacity_reuse(q)
    assert len(q['jobs']) == 10 and q['expected_full_training_optimizer_updates'] == 750000


def test_recovery_rejects_a_smaller_model_or_weaker_emergency_guard():
    q = json.loads(Path('configs/experiments/fm_ls4_native_physionet_queue.v3.json').read_text())
    for change in ('model', 'seed', 'guard', 'admission'):
        bad = copy.deepcopy(q)
        if change == 'model':
            bad['jobs'][0]['model_config'] = 'small.json'
        elif change == 'seed':
            bad['seeds'] = [42]
        elif change == 'guard':
            bad['settings']['emergency_commit_headroom_bytes'] = 1
        else:
            bad['settings']['minimum_commit_headroom_bytes'] = 42 * 1024 ** 3
        with pytest.raises(ValueError):
            verify_capacity_reuse(bad)
