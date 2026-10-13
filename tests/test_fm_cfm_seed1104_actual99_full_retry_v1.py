from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.register_cfm_seed1104_actual99_full_retry_v1 import CANONICAL, validate_retry


def queues():
    original = json.loads(Path('configs/experiments/fm_cfm_ts_original_queue.v1.json').read_text())
    retry = deepcopy(original)
    job = next(j for j in retry['jobs'] if j['id'] == CANONICAL)
    job.update(id=CANONICAL + '__test_retry', recovery_original_job_id=CANONICAL,
               output_directory='experiments/runs/isolated_test_retry/' + CANONICAL,
               artifact_directory='F:/aicoding/IIA_Data/experiments/isolated_test_retry/' + CANONICAL)
    retry['additional_training_seed_slots'] = 0
    return original, retry, job


def test_same_full_original_protocol_in_separate_output():
    original, retry, _ = queues()
    validate_retry(original, retry)


@pytest.mark.parametrize('field,value', [('epochs', 99), ('expected_optimizer_updates', 301),
                                        ('batch_size', 128), ('seed', 1105),
                                        ('test_points_per_trajectory', 100), ('data_sha256', 'incorrect')])
def test_partial_budget_or_changed_science_rejected(field, value):
    original, retry, job = queues()
    job[field] = value
    with pytest.raises(ValueError):
        validate_retry(original, retry)


def test_emergency_guard_cannot_be_relaxed():
    original, retry, _ = queues()
    retry['resource_policy']['emergency_commit_headroom_bytes'] = 1
    with pytest.raises(ValueError):
        validate_retry(original, retry)


def test_original_output_cannot_be_overwritten():
    original, retry, job = queues()
    old = next(j for j in original['jobs'] if j['id'] == CANONICAL)
    job['output_directory'] = old['output_directory']
    with pytest.raises(ValueError):
        validate_retry(original, retry)


def test_missing_original_slot_cannot_be_hidden():
    original, retry, _ = queues()
    retry['jobs'].pop()
    with pytest.raises(ValueError):
        validate_retry(original, retry)


def test_additional_seed_credit_rejected():
    original, retry, _ = queues()
    retry['additional_training_seed_slots'] = 1
    with pytest.raises(ValueError):
        validate_retry(original, retry)
