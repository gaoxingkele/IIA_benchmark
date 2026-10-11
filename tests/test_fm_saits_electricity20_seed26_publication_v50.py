import copy

import pytest

from scripts.flow_matching.publish_saits_electricity20_seed26_v50 import check_delta


def rows():
    return [dict(algorithm='saits',dataset='electricity',task='imputation',protocol='author',
        pattern='paper_mask',missing_ratio='0.2',metric=m,n='0',required_repeats='5',
        mean='',std='',se='',ci95_low='',ci95_high='',paper_mean='0.5',
        author_equivalence_certified='False',strict_TAB_result='',captured_utc='old')
        for m in ('mae','rmse','mre')]


def advanced(before):
    value=copy.deepcopy(before)
    for row in value:
        row.update(n='1',mean='0.8',strict_TAB_result='False',captured_utc='new')
    return value


def test_exact_one_seed_advance_keeps_all_other_obligations():
    before=rows()+[dict(rows()[0],algorithm='unrelated')]
    after=advanced(before[:3])+copy.deepcopy(before[3:])
    assert check_delta(before,after)==[0,1,2]


@pytest.mark.parametrize('key,value',[
    ('n','5'),('required_repeats','1'),('missing_ratio','0.1'),('algorithm','brits'),
    ('paper_mean','0.7'),('std','0'),('se','0'),('ci95_low','0.8'),
    ('author_equivalence_certified','True'),('strict_TAB_result','True'),('mean','nan')])
def test_invalid_completion_uncertainty_or_paper_promotion_is_rejected(key,value):
    before=rows();after=advanced(before);after[0][key]=value
    with pytest.raises(ValueError):check_delta(before,after)


def test_dropped_or_unchanged_target_metric_is_rejected():
    before=rows();after=advanced(before)
    with pytest.raises(ValueError):check_delta(before,after[:-1])
    after[0]=before[0]
    with pytest.raises(ValueError):check_delta(before,after)
