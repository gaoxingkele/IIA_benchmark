import ast
from copy import deepcopy

import numpy as np
import pytest

from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v2 as old
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v3 as s


def test_same_positive_gene_support_can_have_different_expression_identity():
    columns = {0: (np.array([8, 2]), np.array([1., 2.])),
               1: (np.array([8, 2]), np.array([5., 3.]))}
    with pytest.raises(ValueError):
        s.unique_anchor_candidate(np.array([0, 1]), np.array([2, 2]), {8, 2},
                                  lambda f: set(columns[f][0]))
    # Values come from the fixed transform; no fitted or relaxed tolerance.
    assert s.unique_expression_candidate(np.array([0, 1]), np.array([2., 3.]),
        np.array([2, 2]), {8, 2}, columns.__getitem__, dict(rtol=2e-6, atol=2e-6)) == 2


def test_later_expression_anchor_disambiguates_tied_counts():
    columns = {0: (np.array([8, 2]), np.array([2., 2.])),
               1: (np.array([8, 2]), np.array([5., 3.]))}
    assert s.unique_expression_candidate(np.array([0, 1]), np.array([2., 3.]),
        np.array([2, 2]), {8, 2}, columns.__getitem__, dict(rtol=2e-6, atol=2e-6)) == 2


@pytest.mark.parametrize('values', [[2., 3.], [7., 9.]])
def test_identical_or_contradictory_profiles_are_not_arbitrarily_assigned(values):
    columns = {0: (np.array([8, 2]), np.array([2., 2.])),
               1: (np.array([8, 2]), np.array([3., 3.]))}
    diagnostic = []
    with pytest.raises(ValueError):
        s.unique_expression_candidate(np.array([0, 1]), np.array(values), np.array([2, 2]),
            {8, 2}, columns.__getitem__, dict(rtol=2e-6, atol=2e-6), diagnostic.append)
    assert len(diagnostic) == 1 and diagnostic[0]['expression_candidates'] in [0, 2]


def test_expression_anchor_is_still_not_a_complete_matrix_proof():
    columns = {0: (np.array([8, 2]), np.array([1., 2.])),
               1: (np.array([8, 2]), np.array([5., 3.]))}
    row = s.unique_expression_candidate(np.array([0, 1]), np.array([2., 3.]),
        np.array([2, 2]), {8, 2}, columns.__getitem__, dict(rtol=2e-6, atol=2e-6))
    assert row == 2
    with pytest.raises(ValueError):
        s.check_column_identity(np.array([row]), np.array([10.]), np.array([row]),
                                np.array([11.]), dict(rtol=2e-6, atol=2e-6))


def test_all_matrix_coordinates_pca_whitening_and_original_guards_retained():
    def tree(module):
        return ast.parse((s.ROOT / (module.MODULE.replace('.', '/') + '.py')).read_text(encoding='utf-8'))
    before, after = [{n.name: ast.dump(n) for n in tree(m).body if isinstance(n, ast.FunctionDef)}
                     for m in [old, s]]
    for name in ['column', 'transformed', 'check_column_identity', 'centered_pca',
                 'author_whitening', 'validate_cohort_roster', 'validate_reference_injection']:
        assert before[name] == after[name]
    worker = next(n for n in tree(s).body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = [n.func.id for n in ast.walk(worker) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'unique_expression_candidate' in calls and 'check_column_identity' in calls
    assert 'validate_reference_injection' in calls and 'centered_pca' in calls
    c, previous = s.registration(), old.registration()
    for key in ['inputs', 'expected_reference_cells', 'expected_public_reference_candidates',
        'expected_paper_cells', 'expected_features', 'days', 'encoding_scale', 'encoding_tolerance',
        'pca_dimensions', 'pca_initial_vector_seed', 'pca_arpack_tolerance', 'settings',
        'resource_lock', 'stable_samples', 'numerical_threads']:
        assert c[key] == previous[key]
    assert c['output_directory'] != previous['output_directory']
    guard = s.read(s.ROOT / previous['state_root'] / 'guard_receipt.json')
    assert guard['exit_code'] == 1 and not guard['passed'] and not guard['resource_guard_aborted']


@pytest.mark.parametrize('fault', ['cells', 'features', 'tolerance', 'guard'])
def test_no_scope_reduction_or_tolerance_widening(fault):
    c = deepcopy(s.registration())
    if fault == 'cells': c['expected_paper_cells'] = 10
    elif fault == 'features': c['expected_features'] = 10
    elif fault == 'tolerance': c['encoding_tolerance']['atol'] = 1.
    else: c['settings']['minimum_commit_headroom_bytes'] = 1
    with pytest.raises(ValueError):
        s.verify(c)
