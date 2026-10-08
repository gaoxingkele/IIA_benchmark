import json
from pathlib import Path

import pytest
from scripts.flow_matching import summarize_tsad_execution as summary


def record(run_id,seed,value):
    return {'id':run_id,'model_config':'models/method.json','dataset':'dataset','seed':seed,
            'VUS_entity_macro':{'VUS_ROC':{'mean_over_defined_entities':value}},
            'strict_affiliation_entity_macro':{'1':{'affiliation_f':{'mean_over_defined_entities':None}}}}


def snapshots(monkeypatch,groups):
    monkeypatch.setattr(summary,'_one_range_snapshot',lambda root,project,path: {
        'records':groups[path],'process_observation':{},'seed_aggregates':[]})
    return {'tab_range_execution_config':'original','additional_tab_range_execution_configs':['recovery']}


def test_range_recovery_occupies_one_selected_seed_not_an_extra_repeat(monkeypatch):
    project=snapshots(monkeypatch,{'original':[record('old',1,.2)],'recovery':[record('retry',1,.8)]})
    value=summary.range_snapshot(Path('.'),project,{'retry'})
    assert value['completed_evaluations']==2 and value['selected_evaluations']==1
    metrics=value['seed_aggregates'][0]['metrics']
    assert metrics['VUS_ROC']['n']==1 and metrics['VUS_ROC']['mean']==.8
    assert 'affiliation_f' not in metrics


def test_range_groups_combine_source_cohorts_instead_of_overwriting(monkeypatch):
    project=snapshots(monkeypatch,{'original':[record('old',1,.8)],'recovery':[record('retry',2,.6)]})
    value=summary.range_snapshot(Path('.'),project,{'old','retry'})
    metrics=value['seed_aggregates'][0]['metrics']['VUS_ROC']
    assert metrics['n']==2 and metrics['mean']==.7


def test_duplicate_selected_seed_is_rejected(monkeypatch):
    project=snapshots(monkeypatch,{'original':[record('old',1,.2)],'recovery':[record('retry',1,.8)]})
    with pytest.raises(ValueError,match='counted twice'):
        summary.range_snapshot(Path('.'),project,{'old','retry'})


def test_recovery_metric_mirror_preserves_complete_job_dictionaries_and_tab_resolution():
    root=summary.ROOT
    recovery=json.loads((root/'configs/experiments/fm_resource_recovery_queue.v3.json').read_text())
    mirror=json.loads((root/'configs/experiments/fm_recovery_tab_mirror.v1.json').read_text())
    assert mirror['jobs']==[c['job'] for c in recovery['jobs'] if c['kind'] in ('strict','industrial')]
    assert len(mirror['jobs'])==6
    config=json.loads((root/'configs/experiments/fm_recovery_tab_ranges.v1.json').read_text())
    parent=json.loads((root/config['parent_config']).read_text())
    for key in ('reference_metric_root','tab_commit','metric_semantics','strict_calibration','tab_calibration',
                'undefined_policy','runtime_resources'):
        assert config[key]==parent[key]
