from copy import deepcopy
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA

from scripts.flow_matching import prepare_otcfm_multiome13176_public_independent_v1 as s


def test_same_original_sparse_centered_pca_matches_independent_full_dense_solution():
    rng = np.random.RandomState(39)
    values = rng.lognormal(size=(37, 12)); values[values < 1] = 0
    pcs, model = s.original.centered_pca(sparse.csc_matrix(values), 5, 0, 0.)
    dense = PCA(n_components=5, svd_solver='full').fit(values)
    assert np.allclose(model['explained_variance'], dense.explained_variance_, rtol=1e-10, atol=1e-10)
    # Author-loader coordinates intentionally use float32. Compare against the
    # independently computed dense result in that same output representation;
    # double-precision Gram products must not hide a changed output dtype.
    expected = dense.transform(values).astype(np.float32)
    assert pcs.dtype == expected.dtype
    assert np.allclose(pcs.astype(np.float64) @ pcs.astype(np.float64).T,
                       expected.astype(np.float64) @ expected.astype(np.float64).T,
                       rtol=1e-12, atol=1e-12)
    assert model['relative_projection_residual'] < 1e-10


@pytest.mark.parametrize('fault', ['partial_genes', 'partial_reference', 'coordinate_mismatch',
    'missing_reference_disclosure', 'unqualified_guard', 'global_donor_substitution', 'hidden_duplicate'])
def test_shared_encoding_evidence_and_failed_historical_identity_cannot_be_misrepresented(fault):
    c = s.registration(); r = s.read(s.ROOT / c['alignment_report']); g = s.read(s.ROOT / c['alignment_guard'])
    s.validate_alignment_evidence(r, g)
    if fault == 'partial_genes': r['full_gene_features'] -= 1
    elif fault == 'partial_reference': r['mapped_rows_full_coordinate_checks'] -= 1
    elif fault == 'coordinate_mismatch': r['mapped_but_coordinate_mismatching_rows'] = [7]
    elif fault == 'missing_reference_disclosure': r['original_full_reference_identity_qualified'] = True
    elif fault == 'unqualified_guard': g['passed'] = False
    elif fault == 'global_donor_substitution': r['fully_matching_global_same_day_rows'] = 1
    else: r['duplicate_mapping_author_rows'] = [5]
    with pytest.raises(ValueError): s.validate_alignment_evidence(r, g)


@pytest.mark.parametrize('fault', ['missing_cell', 'duplicate_cell', 'missing_day', 'day_length',
                                  'missing_gene', 'duplicate_gene', 'false_original_track'])
def test_full_target_roster_is_never_filtered_or_relabeled_as_original(fault):
    c = dict(data_track=s.TRACK, expected_cells=8, expected_features=5, days=[2, 3, 4, 7])
    rows, days, features = np.arange(8), np.array([2, 2, 3, 3, 4, 4, 7, 7]), np.arange(5)
    s.validate_target(c, rows, days, features)
    if fault == 'missing_cell': rows = rows[:-1]
    elif fault == 'duplicate_cell': rows[-1] = rows[0]
    elif fault == 'missing_day': days[-2:] = 4
    elif fault == 'day_length': days = days[::2]
    elif fault == 'missing_gene': features = features[:-1]
    elif fault == 'duplicate_gene': features[-1] = features[0]
    else: c['data_track'] = 'historical_original'
    with pytest.raises(ValueError): s.validate_target(c, rows, days, features)


def test_real_all_target_cells_features_and_fixed_day_counts():
    c = s.registration()
    with h5py.File(s.ROOT / c['input']['path'], 'r') as f:
        donor = s.original.column(f['obs'], 'donor').astype(str)
        day = s.original.column(f['obs'], 'day').astype(int)
        rows = np.flatnonzero((donor == '13176') & np.isin(day, c['days']))
        features = s.original.column(f['var'], f['var'].attrs['_index'])
        s.validate_target(c, rows, day[rows], features)
        assert dict(zip(*np.unique(day[rows], return_counts=True))) == {2: 9546, 3: 8965, 4: 9407, 7: 7466}


@pytest.mark.parametrize('field', ['expected_cells', 'expected_features', 'pca_dimensions',
    'encoding_scale', 'historical_paper_artifact_identity_certified', 'settings'])
def test_registration_preserves_whole_cohort_math_and_disclosure(field):
    c = deepcopy(s.registration())
    if field == 'settings': c[field]['minimum_commit_headroom_bytes'] = 1
    elif field == 'historical_paper_artifact_identity_certified': c[field] = True
    else: c[field] = 1
    with pytest.raises(ValueError): s.verify(c)


def test_independent_registration_is_cold_and_keeps_exact_reference_failure():
    code = ('from scripts.flow_matching import prepare_otcfm_multiome13176_public_independent_v1 as s;'
        'import sys,json;c=s.registration();s.verify(c);'
        'assert not any(k in sys.modules for k in ["numpy","scipy","torch"]);'
        'assert c["strict_full_reference_identity_gate_passed"] is False;'
        'assert c["historical_paper_artifact_identity_certified"] is False;'
        'assert c["original_author_whiten_flag"] is True;'
        'print(json.dumps([c["data_track"],c["expected_cells"],c["expected_features"]]))')
    p = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == [s.TRACK, 35384, 23418]
