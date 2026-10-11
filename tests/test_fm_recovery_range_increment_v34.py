import copy

import pytest

from scripts.flow_matching.publish_recovery_range_increment_v34 import range_only_delta


def fixture():
    before = [dict(task='anomaly_detection', algorithm='registered', dataset='SMAP',
        protocol='validation_only_1_percent_no_PA', metric='VUS_ROC', n='4', required_repeats='5', mean='.5'),
        dict(task='anomaly_detection', algorithm='registered', dataset='SMAP',
            protocol='validation_only_1_percent_no_PA', metric='f1', n='5', required_repeats='5', mean='.1')]
    after = [dict(before[0], n='5', mean='.55'), before[1]]
    return before, after


def test_only_complete_original_range_repeat_increment_changes():
    before, after = fixture()
    assert range_only_delta(before, after) == [0]
    wrong = copy.deepcopy(after)
    wrong[1]['mean'] = '.99'
    with pytest.raises(ValueError, match='range rows'):
        range_only_delta(before, wrong)


@pytest.mark.parametrize('n', ['3', '4', '6'])
def test_no_repeat_disappearance_same_count_rewrite_or_extra_seed(n):
    before, after = fixture()
    after[0]['n'] = n
    with pytest.raises(ValueError, match='repeat budget'):
        range_only_delta(before, after)


def test_original_metric_identities_and_author_rows_remain():
    before, after = fixture()
    with pytest.raises(ValueError, match='slot'):
        range_only_delta(before, after[:-1])
    wrong = copy.deepcopy(after)
    wrong[0]['protocol'] = 'author_threshold_PA'
    with pytest.raises(ValueError):
        range_only_delta(before, wrong)
