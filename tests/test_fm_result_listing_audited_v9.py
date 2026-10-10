import copy

import pytest

from scripts.flow_matching.publish_result_listing_audited_v9 import merge_audited_rows


def row(**updates):
    result = dict(task='anomaly_detection', algorithm='pi', dataset='PSM',
                  protocol='author_threshold_PA', metric='f1', n=1,
                  required_repeats=6, mean=0.97, std=None, se=None,
                  ci95_low=None, ci95_high=None, captured_utc='2026-10-10T17:00:00+00:00',
                  author_equivalence_certified=False)
    result.update(updates)
    return result


def update(**overrides):
    values = dict(n=2, std=0.0, se=0.0, ci95_low=0.97, ci95_high=0.97,
                  captured_utc='2026-10-10T18:00:00+00:00')
    values.update(overrides)
    return row(**values)


def test_merge_replaces_partial_group_without_mutating_history_or_missing_values():
    base = [row(), row(algorithm='pending', n=0, mean=None)]
    before = copy.deepcopy(base)
    merged = merge_audited_rows(base, [update()])
    assert base == before and len(merged) == len(base)
    assert merged[0]['n'] == 2 and merged[0]['required_repeats'] == 6
    assert merged[1] == base[1] and merged[1]['mean'] is None


@pytest.mark.parametrize('overrides,reason', [
    ({'protocol': 'strict_TAB'}, 'canonical slot'),
    ({'required_repeats': 5}, 'repeat budget'),
    ({'captured_utc': '2026-10-10T16:00:00+00:00'}, 'not newer'),
    ({'author_equivalence_certified': True}, 'equivalence'),
])
def test_merge_rejects_protocol_budget_timestamp_or_equivalence_drift(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        merge_audited_rows([row()], [update(**overrides)])


def test_duplicate_metric_does_not_count_as_another_training_seed():
    with pytest.raises(ValueError, match='Duplicate'):
        merge_audited_rows([row()], [update(), update()])
