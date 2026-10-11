"""Check metric recomputation on independent hand-computable point predictions."""
import numpy as np
import pytest

from scripts.flow_matching.audit_grasp_complete_replay_union_v1 import metrics


def test_validation_only_threshold_is_not_selected_from_test_labels():
    validation = np.array([0., 1., 2., 3.])
    scores = np.array([0., 4., 5., 2.])
    labels = np.array([0, 1, 0, 1])
    result = metrics(labels, scores, validation)
    fixed = result['validation_only_control']
    assert fixed['threshold'] == pytest.approx(2.97)
    assert fixed['precision'] == fixed['recall'] == fixed['f1'] == .5
    changed = metrics(1-labels, scores, validation)
    assert changed['validation_only_control']['threshold'] == fixed['threshold']
    assert result['paper_retrospective']['Best_F1'] == pytest.approx(.8)


def test_exact_threshold_equality_remains_negative():
    result = metrics(np.array([0, 1]), np.array([1., 2.]), np.ones(4))
    assert result['validation_only_control']['precision'] == 1.
    assert result['validation_only_control']['recall'] == 1.
    assert result['validation_only_control']['f1'] == 1.
