import math
import pytest
from scripts.flow_matching import audit_tranad_original_SPOT_completed_cases_v1 as s


def test_independent_event_PA_keeps_original_zero_index_quirk():
    assert s.source_pa([0,0,1,0,0],[1,1,1,0,0])==[False,True,True,False,False]
    assert s.source_pa([1,0,0,0,1],[1,1,1,0,1])==[True,True,True,False,True]


def test_source_point_metrics_have_epsilon_and_binary_not_continuous_ROC():
    a=s.counts([1,0,1,0],[1,1,0,0])
    assert a['TP']==a['TN']==a['FP']==a['FN']==1
    assert a['ROC/AUC']==.5 and 0<a['f1']<.5
    assert s.counts([1,0],[1,0])['f1']<1


def test_metric_validation_refuses_fabricated_finite_or_changed_count():
    a=s.counts([1,0],[1,0]);s.compare(a,a)
    bad=dict(a,TP=2)
    with pytest.raises(ValueError):s.compare(bad,a)
    with pytest.raises(ValueError):s.compare(dict(a,f1=None),a)


def test_one_class_source_binary_ROC_is_explicitly_undefined():
    a=s.counts([1,1],[1,1]);assert a['ROC/AUC'] is None
    s.compare(dict(a,**{'ROC/AUC':0}),a)
    with pytest.raises(ValueError):s.compare(dict(a,**{'ROC/AUC':.9}),a)
