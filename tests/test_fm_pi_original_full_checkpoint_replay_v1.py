import numpy as np
import pytest

from scripts.flow_matching.replay_pi_original_full_checkpoint_v1 import compare_complete_streams


def test_full_stream_replay_rejects_dropped_geometry_or_changed_thresholded_labels():
    saved={'test_energy':np.array([1.,2.,3.],dtype=np.float32),'labels':np.array([0,1,1]),'threshold':np.array(1.5)}
    tolerance=dict(rtol=1e-6,atol=1e-7)
    assert compare_complete_streams(saved,saved,tolerance)['test_energy']['bitwise_equal']
    with pytest.raises(ValueError,match='geometry'):
        compare_complete_streams(saved,dict(saved,test_energy=saved['test_energy'][:-1]),tolerance)
    with pytest.raises(ValueError,match='stream differs'):
        compare_complete_streams(saved,dict(saved,labels=np.array([1,1,1])),tolerance)
    with pytest.raises(ValueError,match='stream differs'):
        compare_complete_streams(saved,dict(saved,threshold=np.array(1.50000001)),tolerance)
    with pytest.raises(ValueError,match='finite'):
        compare_complete_streams(saved,dict(saved,test_energy=np.array([1.,2.,np.nan])),tolerance)


def test_replay_requires_all_saved_streams_and_existing_strict_score_tolerance():
    saved={'train_energy':np.array([.1,.2]),'fused_train':np.array([2.,3.])}
    with pytest.raises(ValueError,match='streams required'):
        compare_complete_streams(saved,{'train_energy':saved['train_energy']},dict(rtol=1e-6,atol=1e-7))
    with pytest.raises(ValueError,match='stream differs'):
        compare_complete_streams(saved,dict(saved,fused_train=np.array([2.,3.1])),dict(rtol=1e-6,atol=1e-7))
