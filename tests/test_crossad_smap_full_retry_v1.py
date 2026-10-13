"""Full SMAP recipe, weight-only recovery boundary and original queue fidelity."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from scripts.flow_matching import crossad_smap1103_full_retry_v1 as retry

CONFIG = 'configs/runtime/fm_crossad_smap1103_full20_retry.v1.json'


def original_queue():
    return retry.previous.read(retry.ROOT / retry.previous.ORIGINAL_QUEUE)


def test_all139_slots_unchanged_except_one_fresh_output():
    original = original_queue(); actual = retry.build_queue(original); expected = deepcopy(original)
    next(j for j in expected['jobs'] if j['id'] == retry.CANONICAL)['output_directory'] = retry.STATE + '/jobs/' + retry.CANONICAL
    assert actual == expected
    assert len(actual['jobs']) == 139


@pytest.mark.parametrize('field,value', [('train_epochs', 19), ('batch_size', 64), ('patience', 2)])
def test_full_original_recipe_required(field, value):
    original = original_queue()
    next(j for j in original['jobs'] if j['id'] == retry.CANONICAL)['train_parameters'][field] = value
    with pytest.raises(ValueError, match='training recipe'):
        retry.build_queue(original)


def test_actual_six_epoch_weight_only_state_is_preserved():
    config = retry.previous.read(retry.ROOT / CONFIG)
    trace = retry.verify_incident(retry.ROOT, config['incident'], check_live=False)
    assert trace['epochs_finished'] == 6
    assert config['incident']['optimizer_rng_resume_available'] is False
    assert config['incident']['completed_budget_credit_for_fresh_retry'] == 0
    assert retry.weight_only_save_contract(retry.ROOT / config['incident']['weight_save_source'])


def test_claimed_partial_epoch_credit_rejected():
    config = retry.previous.read(retry.ROOT / CONFIG)
    incident = deepcopy(config['incident']);incident['completed_budget_credit_for_fresh_retry'] = 6
    with pytest.raises(ValueError, match='zero fresh budget'):
        retry.verify_incident(retry.ROOT, incident, check_live=False)


def test_budget_guards_and_actual_native_preflight_verified():
    config = retry.previous.read(retry.ROOT / CONFIG)
    queue, job, _ = retry.verify(config, check_live=False)
    original = original_queue()
    assert config['settings'] == dict(original['settings'], minimum_commit_headroom_bytes=40 * 1024**3)
    assert job['train_parameters'] == next(j for j in original['jobs'] if j['id'] == retry.CANONICAL)['train_parameters']
    assert len(queue['jobs']) == 139


def test_live_same_canonical_worker_rejected():
    config = retry.previous.read(retry.ROOT / CONFIG)
    worker = type('Worker', (), {'info': dict(name='python.exe', create_time=1791905617.644876,
        cmdline=config['incident']['stopped_scientific_identity']['command'])})()
    with patch('psutil.process_iter', return_value=[worker]):
        with pytest.raises(ValueError, match='live canonical'):
            retry.verify_incident(retry.ROOT, config['incident'])
