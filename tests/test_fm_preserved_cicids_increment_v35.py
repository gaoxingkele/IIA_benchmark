import copy

import pytest

from scripts.flow_matching.publish_preserved_cicids_increment_v35 import original, ALGORITHM, complete_original_seed_delta


def fixture():
    before=[dict(task='anomaly_detection',algorithm=ALGORITHM,dataset='CICIDS',
        protocol=original.PROTOCOL,metric=m,n='4',required_repeats='5',mean='.3')
        for m in original.POINT_METRICS+original.RANGE_METRICS]
    after=[dict(r,n='5',mean='.4') for r in before]
    return before,after


def test_every_original_point_and_range_metric_must_complete():
    before,after=fixture()
    assert complete_original_seed_delta(before,after)==list(range(8))
    after[-1]=before[-1]
    with pytest.raises(ValueError,match='eight'):
        complete_original_seed_delta(before,after)


@pytest.mark.parametrize('field,value',[('n','4'),('n','6'),('required_repeats','6'),
    ('dataset','SMAP'),('protocol','author_threshold_PA'),('algorithm','smaller_model')])
def test_no_seed_budget_protocol_or_model_substitution(field,value):
    before,after=fixture()
    after[0][field]=value
    with pytest.raises(ValueError):
        complete_original_seed_delta(before,after)


def test_other_experiment_rows_cannot_change_or_disappear():
    before,after=fixture()
    unrelated=dict(before[0],dataset='SMD',n='5',mean='.2')
    before.append(unrelated)
    after.append(unrelated)
    assert complete_original_seed_delta(before,after)==list(range(8))
    wrong=copy.deepcopy(after)
    wrong[-1]['mean']='.99'
    with pytest.raises(ValueError):
        complete_original_seed_delta(before,wrong)
    with pytest.raises(ValueError):
        complete_original_seed_delta(before,after[:-1])
