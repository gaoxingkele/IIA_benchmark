import copy

import pytest

from scripts.flow_matching.finalize_full_range_publication_v26 import verify_delta
from scripts.flow_matching.publish_full_range_increment_v25 import PROTOCOL


def pair():
    old=dict(task='anomaly_detection',algorithm='fm_x',dataset='D',protocol=PROTOCOL,metric='VUS_ROC',
             n='0',required_repeats='1',mean='',std='',se='',ci95_low='',ci95_high='',source='old',captured_utc='old',
             author_equivalence_certified='False',strict_TAB_result='')
    new=dict(old,n='1',mean='.5',source='new',captured_utc='new',strict_TAB_result='False')
    return [old],[new]


def test_only_native_range_repeats_change_and_all_other_rows_remain_identical():
    old,new=pair()
    point=dict(new[0],metric='f1',mean='.2',captured_utc='point-original')
    old.append(point);new.append(copy.deepcopy(point))
    value=verify_delta(old,new)
    assert value['changed_metric_rows']==value['newly_measured_metric_rows']==1
    assert value['unchanged_rows']==1


@pytest.mark.parametrize('field,value',[('required_repeats','2'),('n','2'),('protocol','author_threshold_PA'),
    ('author_equivalence_certified','True'),('metric','f1'),('strict_TAB_result','True')])
def test_changed_budget_protocol_or_certification_cannot_be_published(field,value):
    old,new=pair();new[0][field]=value
    with pytest.raises(ValueError):verify_delta(old,new)


def test_other_metric_capture_time_cannot_be_refreshed():
    old,new=pair();point=dict(new[0],metric='f1',mean='.2',captured_utc='original')
    old.append(point);new.append(dict(point,captured_utc='new'))
    with pytest.raises(ValueError,match='Only original range'):verify_delta(old,new)


def test_registered_scope_cannot_be_shrunk():
    old,new=pair()
    with pytest.raises(ValueError,match='scope'):verify_delta(old,[])
