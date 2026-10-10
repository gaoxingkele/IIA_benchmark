import copy

import pytest

from scripts.flow_matching import publish_result_listing_native_v8 as publisher


def test_native_giflow_is_imputation_and_old_rows_removed(monkeypatch):
    report=dict(native_summary=[dict(algorithm='giflow'),dict(algorithm='another')])
    observed=[]
    def base_capture(value,*args):
        observed.extend(r['algorithm'] for r in value['native_summary'])
        return []
    monkeypatch.setattr(publisher.legacy,'captured_rows',base_capture)
    row=dict(task='imputation',algorithm='GiFlow/main',dataset='air36',protocol='giflow_native/released_mirror',
        metric='native_mean_batch_mae',n=0,required_repeats=5,mean=None,std=None,se=None,ci95_low=None,ci95_high=None)
    native=dict(metric_rows=[row]);before=copy.deepcopy(report)
    result=publisher.merge_rows(report,'base',{},'cfm',[],native)
    assert observed==['another'] and report==before
    assert result==[row] and result[0] is not row
    native['metric_rows'][0]['task']='anomaly_detection'
    with pytest.raises(ValueError,match='task/protocol'):publisher.merge_rows(report,'base',{},'cfm',[],native)
