"""Data-leakage, complete coverage and artifact acceptance tests, not scores."""
import hashlib
import json

import numpy as np
import pytest

from scripts.flow_matching.prepare_tsad_execution import split_and_normalize, load_entities
from scripts.flow_matching.run_tsad_experiment import (
    align_scores, block_f1_interval, covering_windows, event_metrics, evaluate_saved,
)
from scripts.flow_matching.run_tsad_queue import complete_result


RECIPE = {'train_fraction': .8, 'purge_points': 2, 'minimum_segment_points': 3}


def test_preprocessing_uses_only_fit_segment_and_excludes_purge():
    train = np.arange(20.).reshape(-1, 1)
    train[0] = np.nan
    labels = np.array([0, 1, 0, 1])
    first, _, audit = split_and_normalize(train, np.arange(4.).reshape(-1, 1), labels, RECIPE)
    modified = train.copy()
    modified[14:] = 10000.
    second, _, after = split_and_normalize(modified, np.full((4, 1), 99999.), labels, RECIPE)
    np.testing.assert_array_equal(first['train'], second['train'])
    assert audit['fit_interval'] == [0, 14] and audit['purged_interval'] == [14, 16]
    assert audit['mean'] == after['mean'] and audit['median'] == after['median']
    assert audit['repairs']['train'] == 1


@pytest.mark.parametrize('length,window', [(100, 100), (201, 100), (250, 100), (17, 8)])
def test_tail_alignment_gives_one_score_per_original_timestamp(length, window):
    values = np.arange(length, dtype=float)[:, None]
    windows, starts = covering_windows(values, window)
    scores = align_scores(windows[..., 0], starts, length, window)
    np.testing.assert_array_equal(scores, values[:, 0])
    assert starts[-1] == length - window


def test_short_input_and_incomplete_or_nonfinite_scores_fail():
    with pytest.raises(ValueError, match='shorter'):
        covering_windows(np.ones((7, 2)), 8)
    with pytest.raises(ValueError, match='coverage'):
        align_scores(np.ones((1, 8)), np.array([1]), 9, 8)
    with pytest.raises(ValueError, match='nonfinite'):
        align_scores(np.full((1, 8), np.nan), np.array([0]), 8, 8)


def test_event_boundaries_remain_separate_for_diagnostics():
    entities = {'a': (np.array([0, 1, 1]), np.array([0., 0., 2.])),
                'b': (np.array([1, 1, 0]), np.array([0., 0., 0.]))}
    result = event_metrics(entities, 1.)
    assert result['events'] == 2 and result['detected_events'] == 1
    assert result['event_recall'] == .5 and result['mean_detection_delay_points'] == 1


def test_strict_threshold_is_invariant_to_test_scores_and_tab_is_marked_diagnostic():
    settings = {'strict_false_alarm_percent': [1], 'primary_false_alarm_percent': 1,
                'bootstrap_resamples': 20, 'bootstrap_block_points': 2,
                'tab_threshold_ratios_percent': [1, 10], 'tab_commit': 'pinned'}
    truth = np.array([0, 0, 1, 1])
    values = {'a': (truth, np.arange(4.))}
    record = evaluate_saved(values, np.arange(10.), np.arange(10.), 'a' * 64, 'b' * 64, settings, 1)
    altered = evaluate_saved({'a': (truth, np.arange(4.) * 100)}, np.arange(10.), np.arange(10.), 'a' * 64, 'b' * 64, settings, 1)
    assert record['strict']['1']['calibration'] == altered['strict']['1']['calibration']
    assert record['strict']['1']['point_adjustment'] is False
    assert record['tab_core_diagnostic']['test_labels_used_to_select_best_grid'] is True
    assert record['tab_core_diagnostic']['pending_metrics']


def test_bootstrap_is_deterministic_and_clusters_whole_entities():
    entities = {'a': (np.array([0, 1]), np.array([0., 2.])),
                'b': (np.array([0, 1]), np.array([2., 0.]))}
    first = block_f1_interval(entities, 1., 22, 100, 2)
    assert first == block_f1_interval(entities, 1., 22, 100, 2)
    assert first['unit'] == 'entity_cluster' and first['low'] <= first['high']


def test_finished_marker_requires_identity_checkpoint_and_scores(tmp_path):
    job = {'output_directory': 'job', 'id': 'x'}
    assert complete_result(tmp_path, job) is False
    directory = tmp_path / 'job'
    directory.mkdir()
    for name in ('scores.npz', 'model.pt'):
        (directory / name).write_bytes(b'artifact')
    digest = hashlib.sha256(b'artifact').hexdigest()
    record = {'status': 'completed', 'experiment_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
              'scores_sha256': digest, 'checkpoint_sha256': digest}
    (directory / 'result.json').write_text(json.dumps(record))
    assert complete_result(tmp_path, job)
    (directory / 'scores.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        complete_result(tmp_path, job)


def test_missing_entity_test_is_not_silently_discarded(tmp_path):
    (tmp_path / 'x_train.csv').write_text('timestamp,value\n0,1\n1,2\n')
    (tmp_path / 'x_test.csv').write_text('timestamp,value,is_anomaly\n0,1,0\n1,2,1\n')
    (tmp_path / 'orphan_train.csv').write_text('timestamp,value\n0,3\n1,4\n')
    spec = {'loader': 'paired_entity_csv', 'root': '.', 'id': 'fixture', 'excluded_columns': ['timestamp'], 'label_column': 'is_anomaly'}
    with pytest.raises(ValueError, match='Missing entity'):
        list(load_entities(tmp_path, spec))
    pairs = list(load_entities(tmp_path, {**spec, 'known_train_only_entities': ['orphan']}))
    assert len(pairs) == 1 and pairs[0][0] == 'x'


def test_csv_timestamp_and_feature_identity_are_checked(tmp_path):
    (tmp_path / 'x_train.csv').write_text('timestamp,value\n0,1\n1,2\n')
    (tmp_path / 'x_test.csv').write_text('timestamp,value,is_anomaly\n1,1,0\n0,2,1\n')
    spec = {'loader': 'paired_entity_csv', 'root': '.', 'id': 'fixture', 'excluded_columns': ['timestamp'], 'label_column': 'is_anomaly'}
    with pytest.raises(ValueError, match='Nonmonotonic'):
        list(load_entities(tmp_path, spec))
