from copy import deepcopy

import pytest

from scripts.flow_matching import run_pi_causal1105_completed_full_replay_v1 as replay
from tests.test_fm_pi_causal_completed_pair_replay_v1 import verify_fixture


def fixture(monkeypatch):
    config, mapping, bindings = verify_fixture(monkeypatch)
    config['cases'] = config['cases'][:1]
    config['cases'][0]['canonical_id'] = replay.CANONICAL
    config.update(replay.BUDGET)
    job = mapping['original']['jobs'][0]
    job.update(id=replay.CANONICAL, seed=1105)
    job['author_config']['data']['win_size'] = 100
    mapping['native0']['recovery_job_id'] = replay.CANONICAL
    result = bindings[id(mapping['point0'])][2]
    result.update(status='completed', active_attention_layers=1,
                  epochs=[dict(epoch=e, steps=518, early_stop=False) for e in (1, 2, 3)],
                  native_train_points=132481, training_windows=132382,
                  native_test_points=87841, scored_test_points=87800,
                  dropped_test_tail_points=41)
    monkeypatch.setattr(replay, 'read', lambda p: mapping[p])
    return config, mapping, result


def test_full_original_scope_and_budget_accepted(monkeypatch):
    config, _, _ = fixture(monkeypatch)
    queue, _, _ = replay.verify(config)
    assert len(queue['jobs']) == 1140
    assert queue['settings']['seeds'] == [42, 1103, 1104, 1105, 1106, 1107]


@pytest.mark.parametrize('key', list(replay.BUDGET))
def test_shortened_scope_or_extra_training_rejected(monkeypatch, key):
    config, _, _ = fixture(monkeypatch)
    config[key] += 1 if key == 'additional_training_seed_slots' else -1
    with pytest.raises(ValueError):
        replay.verify(config)


@pytest.mark.parametrize('change', ['epochs', 'early_stop', 'train_points', 'test_points',
                                  'tail', 'layers', 'window', 'guard', 'tolerance', 'seed'])
def test_incomplete_result_or_changed_protocol_rejected(monkeypatch, change):
    config, mapping, result = fixture(monkeypatch)
    if change == 'epochs':
        result['epochs'][2]['epoch'] = 2
    elif change == 'early_stop':
        result['epochs'][2]['early_stop'] = True
    elif change == 'train_points':
        result['training_windows'] -= 1
    elif change == 'test_points':
        result['scored_test_points'] -= 1
    elif change == 'tail':
        result['dropped_test_tail_points'] = 0
    elif change == 'layers':
        result['active_attention_layers'] = 2
    elif change == 'window':
        mapping['original']['jobs'][0]['author_config']['data']['win_size'] = 50
    elif change == 'guard':
        mapping['native0']['settings'] = dict(minimum_commit_headroom_bytes=16)
    elif change == 'tolerance':
        mapping['native0']['score_tolerance'] = dict(rtol=.1, atol=.1)
    elif change == 'seed':
        mapping['original']['jobs'][0]['seed'] = 1104
    with pytest.raises(ValueError):
        replay.verify(config)


def test_wrong_canonical_id_rejected(monkeypatch):
    config, _, _ = fixture(monkeypatch)
    changed = deepcopy(config)
    changed['cases'][0]['canonical_id'] = replay.CANONICAL.replace('1105', '1106')
    with pytest.raises(ValueError):
        replay.verify(changed)
