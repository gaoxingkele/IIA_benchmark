import copy

import numpy as np
import pytest

from scripts.flow_matching.publish_full_range_increment_v25 import METRICS, PROTOCOL, entity_buffers, range_updates, verify_macros, write_increment_proof
from scripts.flow_matching.publish_result_listing_audited_v9 import merge_audited_rows


def evaluation(seed=1):
    return dict(model_config='configs/models/fm_x.json',dataset='d',seed=seed,
        VUS_entity_macro={m:dict(mean_over_defined_entities=.2*seed) for m in ['VUS_ROC','VUS_PR']},
        strict_affiliation_entity_macro={'1':{'affiliation_f':dict(mean_over_defined_entities=.3*seed)}},
        tab_ratio_affiliation_entity_macro={'affiliation_f':dict(mean_over_defined_entities=.999)})


def rows(required=2):
    return [dict(task='anomaly_detection',algorithm='fm_x',dataset='D',protocol=PROTOCOL,metric=m,
                 n=0,required_repeats=required,mean='',std='',se='',ci95_low='',ci95_high='',
                 source='old',captured_utc='2026-10-10T00:00:00+00:00',author_equivalence_certified='False') for m in METRICS]


def test_only_range_slots_change_and_diagnostic_affiliation_is_excluded():
    original=rows()
    point=dict(original[0],metric='f1',n=2,mean=.123,std=.01,se=.007,ci95_low=.1,ci95_high=.2)
    author=dict(point,protocol='author_threshold_PA',mean=.99)
    original.extend([point,author]);before=copy.deepcopy(original)
    updates=range_updates(original,[evaluation(1),evaluation(2)],'proof','2026-10-11T00:00:00+00:00')
    merged=merge_audited_rows(original,updates)
    assert len(updates)==3 and original==before
    assert merged[-2:]==original[-2:]
    assert [r['n'] for r in updates]==[2,2,2]
    assert updates[-1]['mean']==pytest.approx(.45)
    assert updates[-1]['std']==pytest.approx(np.std([.3,.6],ddof=1))
    assert all(r['ci95_low']<r['mean']<r['ci95_high'] for r in updates)
    assert all(not r['author_equivalence_certified'] and not r['strict_TAB_result'] for r in updates)


def test_single_published_weight_does_not_manufacture_seed_uncertainty():
    updates=range_updates(rows(1),[evaluation()],'proof','2026-10-11T00:00:00+00:00')
    assert all(r['n']==1 and all(r[k] is None for k in ('std','se','ci95_low','ci95_high')) for r in updates)


def test_repeated_evaluation_cannot_add_a_training_seed():
    with pytest.raises(ValueError,match='counted twice'):
        range_updates(rows(),[evaluation(),evaluation()],'proof','2026-10-11T00:00:00+00:00')


def test_original_repeat_budget_cannot_increase():
    with pytest.raises(ValueError,match='original budget'):
        range_updates(rows(1),[evaluation(1),evaluation(2)],'proof','2026-10-11T00:00:00+00:00')


def test_old_completed_seed_cannot_disappear():
    original=range_updates(rows(),[evaluation(1),evaluation(2)],'proof','2026-10-11T00:00:00+00:00')
    with pytest.raises(ValueError,match='disappear'):
        range_updates(original,[evaluation(1)],'proof2','2026-10-12T00:00:00+00:00')


def test_same_count_reuse_keeps_old_capture_time_and_disallows_changed_values():
    original=range_updates(rows(),[evaluation(1),evaluation(2)],'proof','2026-10-11T00:00:00+00:00')
    assert range_updates(original,[evaluation(1),evaluation(2)],'proof2','2026-10-12T00:00:00+00:00')==[]
    wrong=evaluation(2);wrong['VUS_entity_macro']['VUS_ROC']['mean_over_defined_entities']=.5
    with pytest.raises(ValueError,match='same-count'):
        range_updates(original,[evaluation(1),wrong],'proof2','2026-10-12T00:00:00+00:00')


def test_new_unknown_algorithm_cannot_create_unregistered_metric_slots():
    wrong=evaluation();wrong['model_config']='configs/models/unknown.json'
    with pytest.raises(ValueError,match='existing canonical'):
        range_updates(rows(),[wrong],'proof','2026-10-11T00:00:00+00:00')


def test_full_buffer_range_keeps_boundary_events_and_truncated_median():
    labels=np.array([1,1,1,0,1,1,1,1])
    value=dict(status='defined',max_buffer=6,buffers_evaluated=7,threshold_samples=250)
    entity_buffers(labels,value)
    for field,changed in [('max_buffer',4),('buffers_evaluated',3),('threshold_samples',100)]:
        wrong=dict(value);wrong[field]=changed
        with pytest.raises(ValueError,match='original integer buffers'):
            entity_buffers(labels,wrong)


def test_undefined_reference_cannot_be_reported_as_zero():
    labels=np.zeros(5)
    value=dict(status='undefined_single_class_reference',VUS_ROC=None,VUS_PR=None)
    entity_buffers(labels,value)
    with pytest.raises(ValueError,match='become zero'):
        entity_buffers(labels,dict(value,VUS_ROC=0))


def test_entity_macro_reports_undefined_counts_without_zero_imputation():
    vus={'a':dict(VUS_ROC=.8,VUS_PR=.6),'b':dict(VUS_ROC=None,VUS_PR=None)}
    aff={'a':{'1':dict(affiliation_f=.5,affiliation_precision=.5,affiliation_recall=.5)},
         'b':{'1':dict(affiliation_f=None,affiliation_precision=None,affiliation_recall=None)}}
    macro=lambda mean:dict(mean_over_defined_entities=mean,defined_entities=1,undefined_entities=1)
    e=dict(VUS_per_entity=vus,VUS_entity_macro=dict(VUS_ROC=macro(.8),VUS_PR=macro(.6)),
           strict_affiliation_per_entity=aff,strict_affiliation_entity_macro={'1':{k:macro(.5) for k in ['affiliation_f','affiliation_precision','affiliation_recall']}})
    verify_macros(e)
    wrong=copy.deepcopy(e);wrong['VUS_entity_macro']['VUS_ROC']['mean_over_defined_entities']=.4
    with pytest.raises(ValueError,match='macro differs'):verify_macros(wrong)
    wrong=copy.deepcopy(e);wrong['VUS_entity_macro']['VUS_ROC']['undefined_entities']=0
    with pytest.raises(ValueError,match='counts'):verify_macros(wrong)


def test_new_proof_directory_is_created_without_overwriting_prior_evidence(tmp_path):
    import json
    path=tmp_path/'new-report'/'proof.json'
    value=dict(all_original_experiments_complete=False,new_training_seed_slots=0)
    write_increment_proof(path,value)
    assert json.loads(path.read_text(encoding='utf-8'))==value
    before=path.read_bytes()
    with pytest.raises(FileExistsError):write_increment_proof(path,dict(value,new_training_seed_slots=1))
    assert path.read_bytes()==before
