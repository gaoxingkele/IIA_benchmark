import numpy as np
import pytest
from scripts.flow_matching import audit_pi_closed_score_protocol_controls_v1 as current


def test_original_PA_changes_only_predictions_on_detected_positive_segments():
    labels=np.array([0,1,1,1,0,0])
    actual=current.protocol_controls(np.array([0.,1.,2.,3.,4.]),np.array([0.,0.,0.,10.,0.,2.]),labels,20.)
    raw=next(r for r in actual['controls'] if r['calibration']=='author_pooled_train_test' and not r['label_dependent_PA'])
    PA=next(r for r in actual['controls'] if r['calibration']=='author_pooled_train_test' and r['label_dependent_PA'])
    assert raw['confusion']==dict(tp=1,fp=0,fn=2,tn=3)
    assert PA['confusion']==dict(tp=3,fp=0,fn=0,tn=3)
    assert raw['threshold']==PA['threshold']
    assert raw['metrics']['f1']==.5 and PA['metrics']['f1']==1.


def test_no_label_calibration_and_train_only_threshold_does_not_use_test_distribution():
    train=np.array([0.,1.,2.,3.,4.]);scores=np.array([0.,10.,5.,6.,0.,2.]);labels=np.array([0,1,1,1,0,0])
    first=current.protocol_controls(train,scores,labels,20.)
    label_permuted=current.protocol_controls(train,scores,labels[::-1],20.)
    shifted=current.protocol_controls(train,scores+100,labels,20.)
    assert first['thresholds']==label_permuted['thresholds']
    assert first['thresholds']['train_only_same_percentile']==shifted['thresholds']['train_only_same_percentile']
    assert first['thresholds']['author_pooled_train_test']!=shifted['thresholds']['author_pooled_train_test']
    assert all(not r['strict_TAB_result'] for r in first['controls'])


@pytest.mark.parametrize('ratio',[-1,0,100,101])
def test_invalid_percentile_cannot_be_silently_selected_from_test_scores(ratio):
    with pytest.raises(ValueError):
        current.protocol_controls(np.array([1.,2.]),np.array([1.,2.]),np.array([0,1]),ratio)
