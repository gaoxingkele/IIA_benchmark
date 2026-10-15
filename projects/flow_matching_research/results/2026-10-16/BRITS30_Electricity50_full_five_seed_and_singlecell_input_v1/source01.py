from copy import deepcopy
import json
import subprocess
import sys
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v1 as s


def test_centered_sparse_pca_matches_independent_full_dense_svd():
    rng=np.random.RandomState(7);values=rng.normal(size=(43,11))+np.arange(11)[None,:]*7
    values[np.abs(values)<.3]=0
    matrix=sparse.csc_matrix(values)
    pcs,model=s.centered_pca(matrix,5,0,0.)
    dense=PCA(n_components=5,svd_solver='full').fit(values)
    assert np.allclose(model['explained_variance'],dense.explained_variance_,rtol=1e-10,atol=1e-10)
    assert np.allclose(model['components'].T@model['components'],dense.components_.T@dense.components_,rtol=1e-9,atol=1e-9)
    assert np.allclose(pcs@pcs.T,dense.transform(values)@dense.transform(values).T,rtol=1e-5,atol=1e-5)
    assert model['relative_projection_residual']<1e-10
    assert model['orthogonality_error']<1e-10


@pytest.mark.parametrize('dimensions',[0,-1,11,12])
def test_invalid_pca_components_cannot_reduce_or_replace_full_data(dimensions):
    with pytest.raises(ValueError):s.centered_pca(sparse.eye(11,format='csc'),dimensions,0,0.)


def test_exact_original_whitening_block_matches_declared_full_cohort_math():
    values=np.random.RandomState(1).normal(size=(101,5)).astype(np.float32)+np.arange(5,dtype=np.float32)
    actual,scaler=s.author_whitening(values)
    expected=StandardScaler().fit_transform(values)
    assert np.array_equal(actual,expected)
    assert np.array_equal(values,np.random.RandomState(1).normal(size=(101,5)).astype(np.float32)+np.arange(5,dtype=np.float32))
    assert np.allclose(actual.mean(axis=0),0,atol=1e-6) and np.allclose(actual.std(axis=0),1,atol=1e-6)


def test_unique_gene_support_matching_does_not_assume_row_order_or_names():
    rows={0:{2,9},1:{5,9},2:{2,5}}
    assert s.unique_anchor_candidate(np.array([1,0]),np.array([2,2,2]),{2,5,9},rows.__getitem__)==9
    with pytest.raises(ValueError):s.unique_anchor_candidate(np.array([0]),np.array([2,2,2]),{2,5,9},rows.__getitem__)
    # Two rare genes identify a candidate; the complete matrix gate must reject
    # it if a later gene contradicts that candidate. Anchors alone are no proof.
    candidate=s.unique_anchor_candidate(np.array([0,1,2]),np.array([2,2,2]),{2,5,9},rows.__getitem__)
    assert candidate==9
    with pytest.raises(ValueError):
        s.check_column_identity(np.array([candidate]),np.array([1.]),np.array([],dtype=int),np.array([]),dict(rtol=2e-6,atol=2e-6))


@pytest.mark.parametrize('fault',['wrong_coordinate','missing_gene','wrong_value','nonfinite'])
def test_full_matrix_gate_cannot_accept_partial_or_wrong_row_candidates(fault):
    indices=np.array([1,4,7]);truth=np.array([1.,3.,4.]);actual=truth.copy();mapped=indices.copy()
    if fault=='wrong_coordinate':mapped[1]=5
    elif fault=='missing_gene':mapped=mapped[:-1];actual=actual[:-1]
    elif fault=='wrong_value':actual[1]+=1
    else:actual[1]=np.nan
    with pytest.raises(ValueError):s.check_column_identity(indices,truth,mapped,actual,dict(rtol=2e-6,atol=2e-6))


def test_encoding_transcription_preserves_zero_and_known_scale_without_normalized_embedding_reuse():
    counts=np.array([0.,1.,3.,10.]);public=np.log1p(counts*.07)
    actual=s.transformed(public,100.)
    assert np.allclose(actual,np.log1p(counts*7),rtol=1e-14,atol=1e-14)
    assert actual[0]==0


@pytest.mark.parametrize('values,scale',[(np.array([np.nan]),100),(np.array([-1.]),100),(np.array([1.]),0),(np.array([1.]),-1)])
def test_invalid_expression_coordinates_cannot_be_silently_filled(values,scale):
    with pytest.raises(ValueError):s.transformed(values,scale)


def test_categorical_identity_decoding_rejects_missing_codes(tmp_path):
    import h5py
    path=tmp_path/'metadata.h5'
    with h5py.File(path,'w') as f:
        g=f.create_group('donor');g.create_dataset('categories',data=np.array(['32606','13176'],dtype=h5py.string_dtype()));g.create_dataset('codes',data=[1,0,1])
        assert s.column(f,'donor').tolist()==['13176','32606','13176']
        g['codes'][1]=-1
        with pytest.raises(ValueError):s.column(f,'donor')


@pytest.mark.parametrize('fault',['reference_donor','paper_donor','cells','features','scale','PCA','guard'])
def test_frozen_full_scope_and_resource_settings_reject_shortcuts(fault):
    c=s.registration()
    if fault=='reference_donor':c['reference_donor']='13176'
    elif fault=='paper_donor':c['paper_donor']='32606'
    elif fault=='cells':c['expected_paper_cells']=10
    elif fault=='features':c['expected_features']=10
    elif fault=='scale':c['encoding_scale']=1.
    elif fault=='PCA':c['pca_dimensions']=2
    else:c['settings']['emergency_commit_headroom_bytes']=1
    with pytest.raises(ValueError):s.verify(c)


def test_full_registration_does_not_load_numerical_models_in_waiting_controller():
    code='from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v1 as s;import sys,json;c=s.registration();s.verify(c);assert not any(k in sys.modules for k in ["numpy","scipy","torch"]);print(json.dumps(dict(cells=c["expected_paper_cells"],features=c["expected_features"])))'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    assert json.loads(p.stdout)==dict(cells=35384,features=23418)
