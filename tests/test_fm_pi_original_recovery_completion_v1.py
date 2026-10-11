import numpy as np
import pytest

from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import interval_pa, count_metrics, same_numbers
from scripts.flow_matching.run_pi_transformer_job import author_pa


def test_independent_interval_adjustment_preserves_original_index_zero_boundary():
    for labels,raw in [([1,1,1,0,1,1],[0,0,1,0,0,1]),([1,1,0],[1,0,0]),([0,1,1,0,1],[0,0,0,0,1])]:
        labels,raw=np.array(labels),np.array(raw)
        assert np.array_equal(interval_pa(labels,raw),author_pa(labels,raw))
    assert interval_pa(np.array([1,1,1]),np.array([0,0,1])).tolist()==[0,1,1]


def test_confusion_metrics_keep_no_pa_false_positives_and_missed_events():
    actual,counts=count_metrics(np.array([0,1,1,0]),np.array([1,0,1,0]))
    assert counts==dict(tp=1,fp=1,fn=1,tn=1)
    assert actual==dict(accuracy=.5,precision=.5,recall=.5,f1=.5)
    with pytest.raises(ValueError):same_numbers(actual,dict(actual,f1=.99))
    with pytest.raises(ValueError):same_numbers(actual,dict(f1=.5))
