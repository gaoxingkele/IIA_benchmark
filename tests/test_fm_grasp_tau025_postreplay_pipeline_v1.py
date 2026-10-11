from copy import deepcopy
from pathlib import Path

import pytest

from scripts.flow_matching import run_grasp_tau025_postreplay_pipeline_v1 as pipeline


def fixture(monkeypatch):
    root=Path('/fixture')
    config=dict(source_receipts=[dict(path='frozen',sha256='registered')],
        producer_configuration='producer',first_seed_review_proof='first_proof',
        settings=dict(minimum_commit_headroom_bytes=22),resource_lock='original_lock',python='original_python')
    producer=dict(settings=deepcopy(config['settings']),resource_lock='original_lock',python='original_python',
        full_job_count=1,full_results=[dict(canonical_id='grasp_protocol_v3__tau_0p25__swan__swan_sf__seed1104')])
    proof=dict(passed=True,all_metrics_independently_recomputed_from_full_replay=True,
        recipes=[dict(seed=1103)],actual_completed_recipe_count=1)
    monkeypatch.setattr(pipeline,'ROOT',root)
    monkeypatch.setattr(pipeline,'sha',lambda path:'registered')
    monkeypatch.setattr(pipeline,'read',lambda path: {'producer':producer,'first_proof':proof}[path.name])
    return config,producer,proof


def test_only_same_original_producer_and_completed_first_seed_can_enter_postreview(monkeypatch):
    config,producer,_=fixture(monkeypatch)
    assert pipeline.verify_registration(config) is producer
    monkeypatch.setattr(pipeline,'sha',lambda path:'changed')
    with pytest.raises(ValueError,match='sources changed'):pipeline.verify_registration(config)


@pytest.mark.parametrize('field,value',[
    ('settings',dict(minimum_commit_headroom_bytes=1)),('resource_lock','different_lock'),
    ('python','different_environment'),('full_job_count',0),
    ('full_results',[dict(canonical_id='replaced_seed')])])
def test_postreview_refuses_changed_producer_guards_environment_and_original_seed(monkeypatch,field,value):
    config,producer,_=fixture(monkeypatch)
    producer[field]=value
    with pytest.raises(ValueError,match='resource boundaries'):pipeline.verify_registration(config)


@pytest.mark.parametrize('field,value',[
    ('passed',False),('all_metrics_independently_recomputed_from_full_replay',False),
    ('recipes',[dict(seed=1104)]),('actual_completed_recipe_count',0)])
def test_partial_or_relabeled_first_seed_proof_cannot_enter_publication(monkeypatch,field,value):
    config,_,proof=fixture(monkeypatch)
    proof[field]=value
    with pytest.raises(ValueError,match='independent complete replay'):pipeline.verify_registration(config)
