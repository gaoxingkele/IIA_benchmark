from copy import deepcopy
import numpy as np
import pytest

from scripts.flow_matching import package_otcfm_independent_author_input_v1 as s


def fixture():
    config = dict(cells=8, PC_dimensions=5, original_gene_features=7, day_counts={'2': 2, '3': 2, '4': 2, '7': 2})
    payload = dict(pcs=np.arange(40, dtype=np.float32).reshape(8, 5),
        sample_labels=np.array([2, 2, 3, 3, 4, 4, 7, 7]), source_rows=np.arange(8, dtype=np.int64),
        cell_ids=np.array([f'cell{i}' for i in range(8)]), feature_ids=np.array([f'g{i}' for i in range(7)]))
    return config, payload


def test_numerical_and_lineage_payload_roundtrips_without_pickle(tmp_path):
    c, p = fixture(); s.validate_payload(c, p)
    path = tmp_path / 'input.npz'; np.savez_compressed(path, **p)
    with np.load(path, allow_pickle=False) as f:
        assert all(np.array_equal(f[k], value) and not f[k].dtype.hasobject for k, value in p.items())


@pytest.mark.parametrize('fault', ['NaN', 'wrong_PC_count', 'changed_dtype', 'missing_cell',
    'duplicate_ID', 'reordered_rows', 'wrong_days', 'object_ID', 'object_gene', 'duplicate_gene'])
def test_changed_or_incomplete_author_input_is_rejected(fault):
    c, p = fixture()
    if fault == 'NaN': p['pcs'][0, 0] = np.nan
    elif fault == 'wrong_PC_count': p['pcs'] = p['pcs'][:, :-1]
    elif fault == 'changed_dtype': p['pcs'] = p['pcs'].astype(np.float64)
    elif fault == 'missing_cell': p['pcs'] = p['pcs'][:-1]
    elif fault == 'duplicate_ID': p['cell_ids'][-1] = p['cell_ids'][0]
    elif fault == 'reordered_rows': p['source_rows'] = p['source_rows'][::-1]
    elif fault == 'wrong_days': p['sample_labels'][0] = 7
    elif fault == 'object_ID': p['cell_ids'] = p['cell_ids'].astype(object)
    elif fault == 'object_gene': p['feature_ids'] = p['feature_ids'].astype(object)
    else: p['feature_ids'][-1] = p['feature_ids'][0]
    with pytest.raises(ValueError): s.validate_payload(c, p)


@pytest.mark.parametrize('field', ['cells', 'PC_dimensions', 'author_whiten', 'input_PCs_already_whitened',
                                  'strict_full_reference_identity_gate_passed'])
def test_frozen_complete_input_and_reference_limitation_are_preserved(field):
    c = deepcopy(s.registration())
    if field in ['cells', 'PC_dimensions']: c[field] = 1
    else: c[field] = not c[field]
    with pytest.raises(ValueError): s.verify(c)
