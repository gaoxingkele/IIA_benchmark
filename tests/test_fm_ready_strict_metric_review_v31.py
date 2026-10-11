import copy

import pytest

from scripts.flow_matching.review_ready_strict_metrics_v31 import PROTOCOL,review_delta,stats


def fixture():
    before=[dict(task='anomaly_detection',algorithm='registered',dataset='SMD',protocol=PROTOCOL,
        metric='f1',n='0',required_repeats='5',mean='',std='',se='',ci95_low='',ci95_high='',
        source='old',captured_utc='2026-01-01',author_equivalence_certified='False',strict_TAB_result='False')]
    values=[.2,.4,.6];after=[dict(before[0],**stats(values),source='actual',captured_utc='2026-02-01')]
    return before,after,{('registered','SMD','f1'):values}


def test_independent_samples_and_frozen_repeat_budget_required():
    before,after,samples=fixture()
    assert review_delta(before,after,samples)==[0]
    wrong=copy.deepcopy(after);wrong[0]['mean']=.99
    with pytest.raises(ValueError):review_delta(before,wrong,samples)
    wrong=copy.deepcopy(after);wrong[0]['required_repeats']='6'
    with pytest.raises(ValueError):review_delta(before,wrong,samples)


def test_single_evaluation_cannot_gain_uncertainty_or_author_alignment():
    before,_,_=fixture();values=[.2]
    after=[dict(before[0],**stats(values))];samples={('registered','SMD','f1'):values}
    assert review_delta(before,after,samples)==[0]
    for field,value in [('std',0.),('ci95_low',.2),('author_equivalence_certified','True')]:
        wrong=copy.deepcopy(after);wrong[0][field]=value
        with pytest.raises(ValueError):review_delta(before,wrong,samples)


def test_unrelated_row_cannot_disappear_or_change():
    before,after,samples=fixture()
    untouched=dict(before[0],protocol='author_threshold_PA',n='1',mean='.99')
    before.append(untouched);after.append(untouched)
    assert review_delta(before,after,samples)==[0]
    with pytest.raises(ValueError):review_delta(before,after[:-1],samples)
    wrong=copy.deepcopy(after);wrong[-1]['mean']='.98'
    with pytest.raises(ValueError):review_delta(before,wrong,samples)
