from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching.reconcile_grasp_tau05_stocks_publications_v49 import completed_dependency
from scripts.flow_matching import reconcile_grasp_tau05_stocks_publications_v49 as runner


def fixture():
    return [dict(exit_code=0,resource_guard_aborted=False,complete_postreplay_publication_passed=True),
        dict(passed=True,both_original_native_full_model_replays_and_independent_reviews_verified=True,
             required_original_training_seeds=10,completed_original_training_seeds=2,additional_training_seed_slots=0),
        dict(passed=True,original_seed_count=2,required_original_seeds=10,changed_rows=6,
             other8620rows_and_capture_times_unchanged=True)]


def test_actual_original_complete_pair_and_full_publication_gate():
    completed_dependency(*fixture())


@pytest.mark.parametrize('index,field,value',[(0,'exit_code',1),(0,'resource_guard_aborted',True),
    (0,'complete_postreplay_publication_passed',False),(1,'passed',False),
    (1,'both_original_native_full_model_replays_and_independent_reviews_verified',False),
    (1,'required_original_training_seeds',2),(1,'completed_original_training_seeds',1),
    (1,'additional_training_seed_slots',1),(2,'passed',False),(2,'original_seed_count',1),
    (2,'required_original_seeds',2),(2,'changed_rows',5),(2,'other8620rows_and_capture_times_unchanged',False)])
def test_incomplete_changed_or_failed_dependency_cannot_publish(index,field,value):
    values=deepcopy(fixture());values[index][field]=value
    with pytest.raises(ValueError,match='must finish'):
        completed_dependency(*values)


def registered(monkeypatch):
    config=dict(source_receipts=[dict(path='pinned',sha256='frozen')],producer_configuration='producer',
        native_replay_configuration='native',base_publication='stocks',
        settings=dict(minimum_commit_headroom_bytes=22),resource_lock='same_native_lane',python='same_native_python')
    producer={k:deepcopy(config[k]) for k in ('settings','resource_lock','python')}
    producer['producer_configuration']='native'
    stocks=dict(passed=True,original_training_seeds=[42,43,44,45,46],required_training_seeds=[42,43,44,45,46],
        degrees_of_freedom=4,other8622rows_and_capture_times_unchanged=True)
    monkeypatch.setattr(runner,'ROOT',Path('/fixture'))
    monkeypatch.setattr(runner,'sha',lambda path:'frozen')
    monkeypatch.setattr(runner,'read',lambda path:producer if path.name=='producer' else stocks)
    return config,producer,stocks


def test_frozen_original_guards_and_complete_stocks_required(monkeypatch):
    config,producer,_=registered(monkeypatch)
    assert runner.registration(config) is producer
    monkeypatch.setattr(runner,'sha',lambda path:'changed')
    with pytest.raises(ValueError,match='registration changed'):
        runner.registration(config)


@pytest.mark.parametrize('field,value',[('settings',dict(minimum_commit_headroom_bytes=1)),
    ('resource_lock','other_lane'),('python','other_environment'),('producer_configuration','other_native_model')])
def test_original_producer_environment_and_resource_limits_preserved(monkeypatch,field,value):
    config,producer,_=registered(monkeypatch);producer[field]=value
    with pytest.raises(ValueError,match='resource/environment'):
        runner.registration(config)


@pytest.mark.parametrize('field,value',[('passed',False),('original_training_seeds',[42,43,44,45]),
    ('required_training_seeds',[42,43,44,45]),('degrees_of_freedom',3),
    ('other8622rows_and_capture_times_unchanged',False)])
def test_no_regression_to_partial_stocks_or_changed_inventory(monkeypatch,field,value):
    config,_,stocks=registered(monkeypatch);stocks[field]=value
    with pytest.raises(ValueError,match='five-generator Stocks'):
        runner.registration(config)
