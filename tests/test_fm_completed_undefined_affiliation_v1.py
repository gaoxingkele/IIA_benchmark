"""Guards against treating undefined native TAB results as unrun or zero."""
import hashlib

import numpy as np
import pytest

from scripts.flow_matching import audit_completed_undefined_affiliation_v1 as audit
from scripts.flow_matching.tab_range_metrics import load_reference


@pytest.fixture
def reference():
    return load_reference(audit.ROOT/'experiments/runs/mtsad_protocol_audit/sources/tab/original/ts_benchmark/evaluation/metrics')


def empty_record():
    return dict(status='undefined_reference_nan', affiliation_f=None,
                affiliation_precision=None, affiliation_recall=0.0)


def test_native_reference_no_predictions_preserves_nan_and_zero_recall(reference):
    result = audit.review_empty_prediction(np.array([0, 1, 1, 0, 0]),
                                          np.ones(5), 1.0, empty_record(), reference)
    assert result['full_test_points'] == 5
    assert result['actual_anomaly_points'] == 2
    assert result['actual_reference_events'] == 1
    assert result['predicted_events'] == 0
    assert result['original_TAB_reference_recomputed'] == dict(
        affiliation_f=None, affiliation_precision=None, affiliation_recall=0.0)


@pytest.mark.parametrize('change', ['predictions', 'single_class', 'nan_scores', 'misaligned', 'zero_coercion'])
def test_invalid_undefined_classification_rejected(reference, change):
    labels, scores, record = np.array([0, 1, 1, 0]), np.ones(4), empty_record()
    if change == 'predictions':
        scores[0] = 2
    elif change == 'single_class':
        labels[:] = 0
    elif change == 'nan_scores':
        scores[0] = np.nan
    elif change == 'misaligned':
        scores = scores[:-1]
    else:
        record['affiliation_f'] = 0.0
    with pytest.raises(ValueError):
        audit.review_empty_prediction(labels, scores, 1.0, record, reference)


def calibration(values):
    return dict(calibration_split='validation', false_alarm_percent=1.0,
                rule='validation_score_percentile_linear_strict_greater',
                calibration_points=len(values),
                calibration_sha256=hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest(),
                threshold=float(np.percentile(values, 99, method='linear')))


def test_full_validation_threshold_recomputed():
    values = np.array([1., 7., 3., 8.])
    assert audit.review_validation(values, calibration(values))['exact_validation_threshold_recomputed'] == 7.97


@pytest.mark.parametrize('field,value', [('calibration_split', 'test'), ('threshold', 100.),
                                     ('calibration_points', 3), ('calibration_sha256', '0'*64),
                                     ('false_alarm_percent', 5)])
def test_threshold_identity_changes_rejected(field, value):
    values = np.array([1., 7., 3., 8.])
    saved = calibration(values)
    saved[field] = value
    with pytest.raises(ValueError):
        audit.review_validation(values, saved)


def row():
    return dict(task='anomaly_detection', algorithm='usad', dataset='PRONTO_DAY4',
                protocol=audit.full.PROTOCOL, metric='affiliation_f', n='0', required_repeats='5',
                mean='', std='', se='', ci95_low='', ci95_high='', comparison='', uncertainty_scope='',
                source='original', captured_utc='old')


def records(seeds=range(1103, 1108)):
    return [dict(algorithm='usad', dataset='PRONTO_DAY4', seed=seed,
                 classification='completed_evaluation_undefined_reference_no_predicted_events') for seed in seeds]


def test_only_fully_executed_undefined_cohort_annotated():
    original = row()
    rows = [dict(original)]
    assert audit.annotate_complete_undefined(rows, records(), 'proof', 'new') == [0]
    audit.verify_metadata_delta([original], rows, [0])
    assert rows[0]['mean'] == '' and rows[0]['n'] == '0'
    partial = [row()]
    assert audit.annotate_complete_undefined(partial, records([1103, 1104]), 'proof', 'new') == []
    assert partial == [row()]


def test_duplicate_seed_rejected():
    with pytest.raises(ValueError):
        audit.annotate_complete_undefined([row()], records([1103, 1103]), 'proof', 'new')


def test_five_wrong_seed_identities_rejected():
    with pytest.raises(ValueError):
        audit.annotate_complete_undefined([row()], records(range(1104, 1109)), 'proof', 'new')


@pytest.mark.parametrize('field,value', [('mean', '0'), ('n', '5'), ('required_repeats', '1'),
                                     ('algorithm', 'other'), ('paper_mean', '1')])
def test_metadata_publication_cannot_change_science(field, value):
    original = row()
    updated = dict(original, **{field: value})
    with pytest.raises(ValueError):
        audit.verify_metadata_delta([original], [updated], [0])


def test_unreviewed_metadata_changes_rejected():
    original = [row(), row()]
    updated = [dict(r) for r in original]
    updated[1]['source'] = 'unreviewed'
    with pytest.raises(ValueError):
        audit.verify_metadata_delta(original, updated, [0])


def test_sparse_subset_cannot_replace_or_reduce_historical_statistics():
    original = row()
    original.update(n='5', mean='0.7', std='0.1', se='0.04', ci95_low='0.6', ci95_high='0.8')
    e = dict(model_config='usad.json', dataset='pronto_day4',
             VUS_entity_macro={m:dict(mean_over_defined_entities=None) for m in ('VUS_ROC', 'VUS_PR')},
             strict_affiliation_entity_macro={'1':{'affiliation_f':dict(mean_over_defined_entities=0.6)}})
    result = audit.review_current_numeric_coverage([original], [e])
    assert result['new_numeric_aggregates'] == 0
    assert result['historical_aggregates_with_partial_current_sparse_coverage'][0]['published_defined_values'] == 5
    assert original['n'] == '5' and original['mean'] == '0.7'
    original['n'] = '0'
    with pytest.raises(ValueError):
        audit.review_current_numeric_coverage([original], [e])
