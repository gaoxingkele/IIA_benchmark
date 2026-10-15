from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import audit_otcfm_multiome_full_reference_overlap_v1 as s


def test_sparse_public_gather_matches_independent_full_dense_transform_and_duplicates():
    rows = np.array([0, 2, 5], dtype=np.int32)
    values = np.log1p(np.array([1., 2., 3.]))
    mapping = np.array([5, -1, 0, 2, 5, 4])
    actual, active = s.mapped_public_values(rows, values, mapping, 7, 100.)
    dense = np.zeros(7)
    dense[rows] = np.log1p(np.array([100., 200., 300.]))
    expected = np.zeros(len(mapping))
    expected[active] = dense[mapping[active]]
    assert np.array_equal(active, [0, 2, 3, 4, 5])
    assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual[0] == actual[4] and actual[-1] == 0


def test_gene_comparison_catches_false_positive_and_implicit_zero_not_missing_anchor():
    actual = np.array([0., 8., 4., 5.])
    active = np.array([0, 2, 3])  # row1 remains unresolved, never scored as equal
    bad, error = s.compare_reference_column(np.array([0, 1, 2]), np.array([2., 99., 4.]),
        actual, active, dict(rtol=2e-6, atol=2e-6))
    assert bad.tolist() == [True, False, False, True]
    assert error.tolist() == [2., 0., 0., 5.]


def test_all_sparse_columns_equal_independent_dense_whole_matrix_counterexample():
    truth = np.array([[1., 0., 3.], [0., 2., 0.], [1., 2., 4.]])
    actual = truth.copy(); actual[0, 2] = 0; actual[1, 0] = 7
    mismatches = np.zeros(3, dtype=int)
    errors = np.zeros(3)
    for feature in range(3):
        rows = np.flatnonzero(truth[:, feature])
        bad, error = s.compare_reference_column(rows, truth[rows, feature], actual[:, feature],
                                                np.arange(3), dict(rtol=2e-6, atol=2e-6))
        mismatches += bad; errors = np.maximum(errors, error)
    assert np.array_equal(mismatches, np.count_nonzero(truth != actual, axis=1))
    assert np.array_equal(errors, np.max(np.abs(truth - actual), axis=1))


@pytest.mark.parametrize('rows,mapping', [([0, 0], [0]), ([-1, 2], [0]), ([0, 7], [0]),
                                         ([0, 2], [-2]), ([0, 2], [7])])
def test_invalid_or_duplicate_sparse_indices_cannot_be_hidden(rows, mapping):
    with pytest.raises(ValueError):
        s.mapped_public_values(np.array(rows), np.array([1., 2.]), np.array(mapping), 7, 100.)


def test_global_same_day_expression_candidate_is_separate_from_reference_identity():
    column = lambda feature: (np.array([4, 8]), np.array([3., 7.]))
    args = (np.array([0]), np.array([7.]), np.array([2]))
    tolerance = dict(rtol=2e-6, atol=2e-6)
    assert s.expression_candidates(*args, {4}, column, tolerance) == []
    assert s.expression_candidates(*args, {4, 8}, column, tolerance) == [8]


def test_ambiguous_and_unresolved_outcomes_are_retained_without_arbitrary_choice():
    column = lambda feature: (np.array([4, 8]), np.array([3., 3.]))
    assert s.expression_candidates(np.array([0]), np.array([3.]), np.array([2]), {4, 8},
                                    column, dict(rtol=2e-6, atol=2e-6)) == [4, 8]


@pytest.mark.parametrize('field', ['expected_reference_cells', 'expected_public_reference_candidates',
    'expected_public_cells', 'expected_features', 'encoding_scale', 'encoding_tolerance', 'settings'])
def test_scope_and_fixed_tolerance_cannot_be_reduced_for_a_clean_audit(field):
    c = deepcopy(s.registration())
    if field == 'settings': c[field]['maximum_worker_rss_bytes'] = 1
    elif field == 'encoding_tolerance': c[field]['atol'] = 1.
    else: c[field] = 1
    with pytest.raises(ValueError):
        s.verify(c)


def test_registered_full_audit_is_cold_and_has_no_preprocessing_completion_claim():
    code = ('from scripts.flow_matching import audit_otcfm_multiome_full_reference_overlap_v1 as s;'
        'import sys,json;c=s.registration();s.verify(c);'
        'assert not any(k in sys.modules for k in ["numpy","scipy","torch"]);'
        'print(json.dumps([c["expected_reference_cells"],c["expected_features"],c["settings"]["minimum_commit_headroom_bytes"]]))')
    p = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == [33701, 23418, 32 * 2**30]
