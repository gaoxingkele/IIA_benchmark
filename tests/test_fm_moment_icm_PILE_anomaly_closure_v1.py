import copy
from pathlib import Path
import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as v


def test_independent_NASA_training_pair_and_general_lastrow_are_kept_separate():
    raw=np.arange(1,1001,dtype=np.int64);train=np.arange(1,2001,dtype=np.int64)
    splits,fit=v.expected_splits(dict(family='NASA-MSL'),raw,train)
    assert fit==2000 and len(splits['train'])==1714 and splits['val'][-1]==2000 and splits['test'][-1]==1000
    splits,fit=v.expected_splits(dict(family='Daphnet'),raw)
    assert fit==600 and splits['test'][-1]==999 and len(splits['test'])==299
    with pytest.raises(ValueError):v.expected_splits(dict(family='NASA-MSL'),raw)


@pytest.mark.parametrize('corruption',['none','negative_len','short_gate','raw_identity','fit_extent','mask','nonfinite'])
def test_complete_signal_identity_gate_mask_and_finiteness_are_required(tmp_path,corruption):
    path=tmp_path/'cell';path.mkdir();count=300
    values=np.arange(count,dtype=np.float64);rows=np.arange(1,count+1,dtype=np.int64);mask=np.ones(256)
    names=dict(values='values_T.npy',rows='raw_CSV_record_indices.npy',mask='input_mask_window.npy')
    for key,name in names.items():np.save(path/name,dict(values=values,rows=rows,mask=mask)[key],allow_pickle=False)
    r=dict(time_rows=count,source_dunder_len=45,source_length_gate_selected=True,eligible_stride1_window_count=45,
        source_scaler_fit_rows=180,source_CSV_path='raw',scaler_includes_NASA_validation_rows=False,
        source_general_test_omits_last_parsed_row=True,labels_saved_as_model_inputs=False,
        all_source_signal_coordinates_byte_equal=True,sequence_length=256,stride=1,channels=1,
        source_scaler_mean=[0.],source_scaler_var=[1.],source_scaler_scale=[1.],files={})
    def hash_files():
        for key,name in names.items():
            a=np.load(path/name);r['files'][key]=dict(sha256=v.sha(path/name),shape=list(a.shape),dtype=str(a.dtype))
    hash_files()
    baseline=v.close_split(path,r,rows,180,'Daphnet','raw')
    assert baseline['signal_coordinates']==300 and baseline['all_signal_coordinates_finite']
    if corruption=='none':return
    if corruption=='negative_len':r['source_dunder_len']=-1
    elif corruption=='short_gate':r['source_length_gate_selected']=False
    elif corruption=='fit_extent':r['source_scaler_fit_rows']=300
    elif corruption=='raw_identity':np.save(path/names['rows'],rows[::-1],allow_pickle=False)
    elif corruption=='mask':mask[0]=0.;np.save(path/names['mask'],mask,allow_pickle=False)
    elif corruption=='nonfinite':values[0]=np.nan;np.save(path/names['values'],values,allow_pickle=False)
    hash_files()  # Fresh hashes cannot make wrong arrays or semantics acceptable.
    with pytest.raises(ValueError):v.close_split(path,r,rows,180,'Daphnet','raw')
