"""Full causal PSM1107 audit readiness without modifying any frozen predecessor."""
from copy import deepcopy

import pytest

from scripts.flow_matching import run_pi_causal1107_completed_full_replay_v1 as replay
from tests.test_fm_pi_causal_completed_pair_replay_v1 import verify_fixture


def fixture(monkeypatch):
    config, mapping, bindings = verify_fixture(monkeypatch)
    config['cases'] = config['cases'][:1]
    config['cases'][0]['canonical_id'] = replay.CANONICAL
    config.update(replay.BUDGET)
    job = mapping['original']['jobs'][0]
    job.update(id=replay.CANONICAL, seed=1107)
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
    changed['cases'][0]['canonical_id'] = replay.CANONICAL.replace('1107', '1106')
    with pytest.raises(ValueError):
        replay.verify(changed)


@pytest.mark.parametrize('change', ['branch','epoch_steps'])
def test_original_branch_and_every_full_epoch_must_survive(monkeypatch, change):
    config,mapping,result=fixture(monkeypatch)
    if change=='branch':mapping['original']['jobs'][0]['branch']='source_runtime'
    else:result['epochs'][0]['steps']=517
    with pytest.raises(ValueError):replay.verify(config)


def ready_fixture(monkeypatch):
    import hashlib
    import json
    from scripts.flow_matching import register_pi_causal1107_completed_full_replay_v1 as registration
    job=dict(id=replay.CANONICAL,dataset='PSM',branch='causal_axis_repaired',seed=1107,
             output_directory='original_causal1107')
    queue=dict(jobs=[job]+[dict(id='other_'+str(i)) for i in range(1139)],
               settings=dict(seeds=[42,1103,1104,1105,1106,1107]))
    result=dict(status='completed',parameter_count=1930289,active_attention_layers=1,
        steps_per_epoch=518,native_train_points=132481,training_windows=132382,
        native_test_points=87841,scored_test_points=87800,dropped_test_tail_points=41,
        epochs=[dict(epoch=i,steps=518,early_stop=False) for i in [1,2,3]],
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
        artifacts=[dict(path='artifact_'+str(i),sha256='a'*64) for i in range(7)])
    monkeypatch.setattr(registration,'read',lambda path:result)
    monkeypatch.setattr(registration,'digest',lambda path:'a'*64)
    return registration,queue,result


def test_completed_original_result_is_required_before_any_registration(monkeypatch):
    registration,queue,result=ready_fixture(monkeypatch)
    assert registration.require_original_complete(queue)[1] is result
    def read(path):
        if path=='configs/experiments/fm_pi_transformer_author_queue.v1.json':return queue
        raise FileNotFoundError(path)
    monkeypatch.setattr(registration,'read',read)
    def no_write(*args,**kwargs):raise AssertionError('Pending source must not create any configuration')
    monkeypatch.setattr(registration,'exclusive_json',no_write)
    with pytest.raises(FileNotFoundError):registration.main()


@pytest.mark.parametrize('change',['partial_epochs','unfinished','short_epoch','missing_artifact','wrong_hash','wrong_identity','duplicate_job','missing_seed'])
def test_source_promotion_rejects_incomplete_or_different_original(monkeypatch,change):
    registration,queue,result=ready_fixture(monkeypatch)
    if change=='partial_epochs':result['epochs'].pop()
    elif change=='unfinished':result['status']='running'
    elif change=='short_epoch':result['epochs'][0]['steps']=517
    elif change=='missing_artifact':result['artifacts'].pop()
    elif change=='wrong_hash':result['artifacts'][0]['sha256']='b'*64
    elif change=='wrong_identity':result['experiment_sha256']='b'*64
    elif change=='duplicate_job':queue['jobs'][-1]['id']=queue['jobs'][-2]['id']
    elif change=='missing_seed':queue['settings']['seeds'].pop()
    with pytest.raises(ValueError):registration.require_original_complete(queue)


def test_aligned_review_preserves_all_numerical_settings_and_original_bytes():
    import json
    from pathlib import Path
    from scripts.flow_matching import register_pi_causal1107_completed_full_replay_v1 as registration
    path=Path('configs/reproducibility/fm_pi_original_completed_full_replay_review.2026-10-11_v1.json')
    before=path.read_bytes();template=json.loads(before)
    adjusted=registration.aligned_review_metadata(template)
    assert path.read_bytes()==before and template==json.loads(before)
    assert replay.CANONICAL in adjusted['boundary']
    assert {k:v for k,v in adjusted.items() if k!='boundary'}=={k:v for k,v in template.items() if k!='boundary'}
    assert adjusted is not template and adjusted['settings'] is not template['settings']
