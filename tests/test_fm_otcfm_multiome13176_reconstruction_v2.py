import ast
from copy import deepcopy
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v1 as old
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v2 as s


def roster():
    c = dict(expected_reference_cells=4, expected_public_reference_candidates=5,
             expected_paper_cells=4, reference_donor='32606', days=[2, 3, 4, 7])
    return c, np.array(['32606'] * 4), np.array([2, 3, 4, 7]), np.arange(5), np.arange(5, 9)


def test_public_superset_does_not_reduce_author_reference_or_target():
    c, donors, days, candidates, target = roster()
    s.validate_cohort_roster(c, donors, days, candidates, target)
    public_days = np.array([2, 2, 3, 4, 7, 2, 3, 4, 7])
    mapping = np.array([1, 2, 3, 4])
    extra = s.validate_reference_injection(mapping, candidates, days, public_days, 4)
    assert extra.tolist() == [0]
    # Candidate identification still cannot substitute for the full value gate.
    with pytest.raises(ValueError):
        s.check_column_identity(np.array([0]), np.array([7.]), np.array([0]),
                                np.array([8.]), dict(rtol=2e-6, atol=2e-6))


@pytest.mark.parametrize('fault', ['missing_author', 'missing_target', 'missing_candidate',
    'wrong_donor', 'wrong_day', 'duplicate_candidate', 'duplicate_target', 'overlap'])
def test_all_original_rosters_remain_mandatory(fault):
    c, donors, days, candidates, target = roster()
    if fault == 'missing_author': donors = donors[:-1]
    elif fault == 'missing_target': target = target[:-1]
    elif fault == 'missing_candidate': candidates = candidates[:-1]
    elif fault == 'wrong_donor': donors[0] = '13176'
    elif fault == 'wrong_day': days[0] = 10
    elif fault == 'duplicate_candidate': candidates[-1] = candidates[0]
    elif fault == 'duplicate_target': target[-1] = target[0]
    elif fault == 'overlap': target[0] = candidates[0]
    with pytest.raises(ValueError):
        s.validate_cohort_roster(c, donors, days, candidates, target)


@pytest.mark.parametrize('mapping,days', [([1, 2, 3], [2, 3, 4]),
    ([1, 1, 3, 4], [2, 3, 4, 7]), ([1, 2, 3, 5], [2, 3, 4, 7]),
    ([1, 2, 3, 4], [2, 3, 4, 10])])
def test_no_missing_duplicate_outside_or_wrong_day_author_identity(mapping, days):
    with pytest.raises(ValueError):
        s.validate_reference_injection(np.array(mapping), np.arange(5), np.array(days),
                                        np.array([2, 2, 3, 4, 7, 7]), 4)


def test_real_entire_metadata_roster_including_three_public_only_candidates():
    c = s.registration()
    with h5py.File(s.ROOT / c['inputs']['author']['path'], 'r') as af, \
            h5py.File(s.ROOT / c['inputs']['public']['path'], 'r') as pf:
        donors = s.column(af['obs'], 'donor').astype(str)
        days = s.column(af['obs'], 'day').astype(int)
        pd = s.column(pf['obs'], 'donor').astype(str)
        pt = s.column(pf['obs'], 'day').astype(int)
        candidates = np.flatnonzero((pd == c['reference_donor']) & np.isin(pt, c['days']))
        target = np.flatnonzero((pd == c['paper_donor']) & np.isin(pt, c['days']))
        s.validate_cohort_roster(c, donors, days, candidates, target)
        assert (len(donors), len(candidates), len(target)) == (33701, 33704, 35384)
        assert np.array_equal(s.column(af['var'], af['var'].attrs['_index']),
                              s.column(pf['var'], pf['var'].attrs['_index']))
        assert np.count_nonzero(pt[candidates] == 2) - np.count_nonzero(days == 2) == 3


def test_original_matrix_math_pca_whitening_encoding_and_guards_unchanged():
    def bodies(module):
        tree = ast.parse((s.ROOT / (module.MODULE.replace('.', '/') + '.py')).read_text(encoding='utf-8'))
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = bodies(old), bodies(s)
    for name in ['column', 'transformed', 'unique_anchor_candidate', 'check_column_identity',
                 'centered_pca', 'author_whitening']:
        assert before[name] == after[name]
    c, previous = s.registration(), old.registration()
    for key in ['inputs', 'expected_reference_cells', 'expected_paper_cells', 'expected_features',
        'days', 'encoding_scale', 'encoding_tolerance', 'pca_dimensions', 'pca_initial_vector_seed',
        'pca_arpack_tolerance', 'settings', 'resource_lock', 'stable_samples', 'numerical_threads']:
        assert c[key] == previous[key]
    assert c['output_directory'] != previous['output_directory']
    assert c['state_root'] != previous['state_root']
    prior = old.read(old.ROOT / previous['state_root'] / 'guard_receipt.json')
    assert prior['exit_code'] == 1 and not prior['resource_guard_aborted'] and not prior['passed']


@pytest.mark.parametrize('fault', ['author', 'target', 'candidates', 'features', 'scale', 'guard'])
def test_registration_cannot_weaken_full_data_or_resource_policy(fault):
    c = deepcopy(s.registration())
    keys = dict(author='expected_reference_cells', target='expected_paper_cells',
                candidates='expected_public_reference_candidates', features='expected_features', scale='encoding_scale')
    if fault == 'guard': c['settings']['minimum_commit_headroom_bytes'] = 1
    else: c[keys[fault]] = 1
    with pytest.raises(ValueError):
        s.verify(c)


def test_waiting_registration_stays_cold():
    code = ('from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v2 as s;'
        'import sys,json;c=s.registration();s.verify(c);'
        'assert not any(k in sys.modules for k in ["numpy","scipy","torch"]);'
        'print(json.dumps([c["expected_reference_cells"],c["expected_public_reference_candidates"],c["expected_paper_cells"]]))')
    p = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == [33701, 33704, 35384]
