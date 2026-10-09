"""Do not double count a retry, replace a score, or mix native protocols."""
import copy
import pytest
from scripts.flow_matching.publish_complete_algorithm_dataset_listing_v3 import update_rows


def fixture():
    rows = [dict(task='anomaly_detection', algorithm='main', dataset='swan', protocol='oracle',
        score_recipe="{'source_samples': 5, 'flow_evaluations': 10}", metric='AP',
        n='1', required_repeats='10', mean='0.0', execution_id='retry', original_canonical_id='original')]
    native = [dict(algorithm='GRASP/main', dataset='SWAN', protocol='oracle', metric='AP',
        source_samples='5', flow_evaluations='10', n='1', required_seeds='10', mean='0.0',
        execution_id='retry', original_canonical_id='original', paper_mean='0.7')]
    return rows, dict(seed_aggregates=[]), native


def test_identical_audited_retry_preserves_one_seed_and_real_zero():
    rows, cfm, native = fixture()
    changed, added = update_rows(rows, cfm, native, 'c', 'n', 'new-time')
    assert (changed, added) == (1, 0)
    assert len(rows) == 1 and rows[0]['n'] == 1 and rows[0]['mean'] == 0.
    assert rows[0]['std'] is None and rows[0]['captured_utc'] == 'new-time'


@pytest.mark.parametrize('field,value', [('mean','0.8'), ('execution_id','better_retry'),
    ('original_canonical_id','another_seed'), ('required_seeds','5'), ('protocol','validation')])
def test_reject_conflicting_measured_slot_or_protocol(field, value):
    rows, cfm, native = fixture()
    changed = copy.deepcopy(native); changed[0][field] = value
    with pytest.raises((ValueError, KeyError)):
        update_rows(rows, cfm, changed, 'c', 'n', 'new-time')


def test_empty_canonical_slot_becomes_one_measurement_without_extra_row():
    rows, cfm, native = fixture()
    rows[0].update(n='0',mean='',execution_id='',original_canonical_id='')
    assert update_rows(rows, cfm, native, 'c','n','now') == (1,1)
    assert len(rows) == 1 and rows[0]['n'] == 1
