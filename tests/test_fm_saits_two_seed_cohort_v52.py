import copy
import math

import pytest

from scripts.flow_matching.publish_saits_two_seed_cohort_v52 import (
    check_delta, check_record, verify_registration, KEYS, IDENTITY, METRICS,
)


def tables():
    old = []
    for m in METRICS:
        row = dict(zip(KEYS, IDENTITY))
        row.update(metric=m, n='1', required_repeats='5', paper_mean='0.5', mean='1',
            std='', se='', ci95_low='', ci95_high='', author_equivalence_certified='False',
            strict_TAB_result='False', source='original_v50_source.json', captured_utc='before',
            uncertainty_scope='one seed', comparison='partial')
        old.append(row)
    old.append(dict(old[0], dataset='other', source='historical_v42_source.json'))
    new = copy.deepcopy(old)
    for row in new[:3]:
        row.update(n='2', mean='2', std=str(math.sqrt(2)), se='1', ci95_low='-10.706204736',
            ci95_high='14.706204736', source='two_seed.json', captured_utc='after',
            uncertainty_scope='two canonical seeds, df1, incomplete', comparison='partial_2of5')
    return old, new


def record_fixture(seed=27):
    wanted = 'saits_paper_saits_electricity_author_missing0.2_seed%d' % seed
    metrics = dict(mae=.8, rmse=1.2, mre=.4)
    completion = dict(full_training_finished=True, benchmark_smoke=False,
        original_maximum_epochs=10000, next_epoch=429, author_patience_remaining=0)
    native = dict(id=wanted, seed=seed, method='saits', dataset='electricity', missing_ratio=.2,
        variant='author', corrected_masking=False, diagnostic=False)
    result = dict(config=native, status='completed', parameters=11511000, test_windows=290,
        target_count=1944331, bootstrap_groups=10, bootstrap_group='month', metrics=metrics)
    record = dict(id=wanted, seed=seed, model_parameters=11511000, full_test_shape=[290,100,370],
        held_out_coordinates=1944331, full_training_completion=completion,
        full_original_checkpoint_and_data_identity_verified=True,
        full_test_grid_and_grouped_statistics_replayed=True, original_metrics=metrics,
        replayed_metrics=dict(metrics), independently_recomputed_metrics=dict(metrics))
    checked = dict(id=wanted, seed=seed, full_test_shape=[290,100,370], held_out_coordinates=1944331,
        months=10, actual_full_checkpoint_reinference_completed=True,
        full_truth_masks_units_groups_equal=True, full_group_statistics_independently_recomputed=True,
        independently_recomputed_metrics=dict(metrics))
    return record, checked, native, result


def test_exact_three_existing_slots_and_all_other_sources_preserved():
    old, new = tables()
    assert check_delta(old, new) == [0,1,2]
    assert new[3] == old[3]


@pytest.mark.parametrize('key,value', [
    ('n','5'), ('required_repeats','2'), ('paper_mean','0.4'),
    ('author_equivalence_certified','True'), ('strict_TAB_result','True'),
    ('mean','nan'), ('std','-1'), ('se','-1'), ('ci95_low','3'), ('dataset','other'),
])
def test_forbid_unsupported_cohort_science_or_numeric_change(key, value):
    old, new = tables(); new[0][key] = value
    with pytest.raises(ValueError):
        check_delta(old, new)


def test_forbid_unrelated_historical_source_mutation():
    old, new = tables(); new[3]['source'] = 'invented_v52_source.json'
    with pytest.raises(ValueError):
        check_delta(old, new)


def test_full_native_seed_fixture_validates():
    check_record(*record_fixture(), 27)


@pytest.mark.parametrize('change', ['wrong_seed', 'subset', 'small_model', 'unfinished', 'bad_metric', 'no_independent'])
def test_reject_incomplete_or_misidentified_original_model(change):
    record, checked, native, result = record_fixture()
    if change == 'wrong_seed': record['seed'] = 26
    if change == 'subset': checked['held_out_coordinates'] = 1
    if change == 'small_model': result['parameters'] = 100
    if change == 'unfinished': record['full_training_completion']['author_patience_remaining'] = 1
    if change == 'bad_metric': checked['independently_recomputed_metrics']['mae'] = .9
    if change == 'no_independent': checked['full_truth_masks_units_groups_equal'] = False
    with pytest.raises(ValueError):
        check_record(record, checked, native, result, 27)


@pytest.mark.parametrize('seeds,evidence', [([26,26],[{},{}]), ([26,27],[{}]), ([26,27],[{},{},{}])])
def test_exact_registered_unique_seeds_and_evidence_required(seeds,evidence):
    with pytest.raises(ValueError):
        verify_registration(dict(original_training_seeds=seeds,
            required_training_seeds=[26,27,28,29,30], model_evidence=evidence))


def test_existing_estimator_reports_wide_untruncated_two_seed_interval():
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
    result = independent_stats([1.,3.])
    assert result['n'] == 2 and result['mean'] == 2
    assert result['std'] == pytest.approx(math.sqrt(2)) and result['se'] == pytest.approx(1)
    assert result['ci95_low'] == pytest.approx(-10.706204736)
    assert result['ci95_high'] == pytest.approx(14.706204736)
