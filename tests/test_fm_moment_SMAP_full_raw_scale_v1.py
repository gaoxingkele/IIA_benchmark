import numpy as np
import pytest
from scripts.flow_matching import audit_moment_SMAP_full_raw_scale_v1 as s


def test_full_splits_keep_purge_and_never_fit_test_or_validation_values():
    train=np.arange(80,dtype=np.float64).reshape(40,2);test=np.full((9,2),10**6,dtype=np.float64)
    recipe=dict(train_fraction=.8,purge_points=3,minimum_segment_points=2)
    arrays,labels,stats=s.reconstructed(train,test,np.zeros(9),recipe)
    assert stats['fit_interval']==[0,29] and stats['purged_interval']==[29,32] and stats['validation_interval']==[32,40]
    assert np.array_equal(stats['mean'],train[:29].mean(0))
    assert arrays['test'].shape==(9,2) and labels.shape==(9,)


def test_exact_constant_correction_changes_only_real_constants_without_an_epsilon():
    train=np.ones((200,3),dtype=np.float64)*.999
    train[:,1]=np.arange(200)/1000;train[:,2]=1.+np.arange(200)*1e-12
    test=train[:20].copy();test[:,0]=.5
    recipe=dict(train_fraction=.8,purge_points=10,minimum_segment_points=2)
    a,labels,old=s.reconstructed(train,test,np.zeros(20),recipe)
    b,_,new=s.reconstructed(train,test,np.zeros(20),recipe,True)
    assert new['exact_constant'].tolist()==[True,False,False]
    assert new['std'][0]==0 and new['mean'][0]==train[0,0]
    assert np.all(b['train'][:,0]==0)
    for role in a:s.require_bytes(a[role][:,1:],b[role][:,1:])


def test_every_last_value_dtype_and_label_must_match():
    a=np.arange(12,dtype=np.float32);s.require_bytes(a,a.copy())
    for changed in [a.astype(np.float64),a[:-1],np.append(a[:-1],np.float32(-1))]:
        with pytest.raises(ValueError):s.require_bytes(a,changed)
