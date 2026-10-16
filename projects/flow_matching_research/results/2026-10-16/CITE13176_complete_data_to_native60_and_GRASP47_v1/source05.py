import numpy as np
import pytest
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v1 as s


def test_all_distinct_source_identifiers_are_mapped_without_row_truncation_or_symbol_guessing():
    assert s.unique_map(['g3','g1','g2'],['g2','g3'],'gene')==[2,0]
    for source,target in [(['x','x'],['x']),(['x'],['x','x']),(['x'],['y'])]:
        with pytest.raises(ValueError):s.unique_map(source,target,'cell')


def test_implicit_zero_mismatch_is_not_hidden_by_sparse_nonzero_comparison():
    a=np.array([[1,0,3],[0,2,0]])
    assert s.compare_counts(a,a.astype(np.float32))==6
    bad=a.copy();bad[0,1]=1
    with pytest.raises(ValueError):s.compare_counts(a,bad)


def test_full_RNA_library_excludes_antibody_counts_without_dropping_RNA_features():
    genes,x,total=s.normalized_raw_RNA_row(np.array([5,1000,2]),np.array([2,1,0]),np.array([0,-1,2]))
    assert genes.tolist()==[0,2] and total==7
    np.testing.assert_allclose(x,np.log1p(np.array([2,5])*1e6/7),rtol=0,atol=0)
    assert 1 not in genes


@pytest.mark.parametrize('counts',[[1,-1],[1,np.nan],[1,1.5],[0,0]])
def test_invalid_counts_cannot_be_repaired_or_silently_filtered(counts):
    with pytest.raises(ValueError):s.normalized_raw_RNA_row(np.array(counts),np.array([0,1]),np.array([0,1]))


def test_candidate_diagnostics_check_zeros_and_never_change_frozen_normalizer():
    counts=np.array([[1,2],[3,0]]);expression=np.log1p(counts*(1e6/counts.sum(axis=1))[:,None])
    r=s.normalization_audit(counts,expression,{'published_gene_library':[3,3],'all_raw_RNA_library':[6,6]},dict(rtol=2e-6,atol=2e-6))
    assert r['published_gene_library']['mismatch_coordinates']==0
    assert r['all_raw_RNA_library']['mismatch_coordinates']==3
    assert all(v['coordinates']==4 for v in r.values())


def test_complete_published_QC_roster_full_raw_gene_axis_and_original_guards_are_preserved():
    c=s.registration();old=s.read(s.ROOT/s.POLICY)
    assert sum(c['expected_QC_cells'].values())==c['expected_QC_total_cells']==29394
    assert c['expected_raw_total_cells']==35261 and c['expected_RNA_features']==58676
    assert c['independent_normalization_library']=='all_raw_RNA_library' and not c['candidate_choice_changes_preparation']
    assert c['settings']==old['settings'] and c['resource_lock']==old['resource_lock']
    assert c['stable_samples']==old['stable_samples'] and c['pca_dimensions']==5
    assert not c['historical_Kaggle_or_paper_processed_identity_certified']
