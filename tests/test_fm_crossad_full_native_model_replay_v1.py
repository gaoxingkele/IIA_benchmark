import copy
import numpy as np
import pytest

from scripts.flow_matching.replay_crossad_full_native_model_v1 import compare_scores,compare_metrics


def test_full_native_tail_and_finite_scores_cannot_be_omitted():
    original={k:np.array([.1,.2,.3],dtype=np.float32) for k in ['train','validation','test']}
    compare_scores(original,copy.deepcopy(original))
    for values in [np.array([.1,.2]),np.array([.1,.2,np.nan]),np.array([.1,.2,.4])]:
        bad=copy.deepcopy(original);bad['test']=values
        with pytest.raises(ValueError):compare_scores(original,bad)


def test_native_range_grid_and_original_spot_threshold_metric_changes_are_rejected():
    original={'_vus':[dict(VUS_ROC=.8,VUS_PR=.3)],'_spot_thres_info':[{'threshold':.1,'hyper-parameter':.1}]}
    compare_metrics(original,copy.deepcopy(original))
    for name,key in [('_vus','VUS_ROC'),('_spot_thres_info','threshold')]:
        bad=copy.deepcopy(original);bad[name][0][key]+=.001
        with pytest.raises(ValueError):compare_metrics(original,bad)
    bad=copy.deepcopy(original);bad.pop('_vus')
    with pytest.raises(ValueError):compare_metrics(original,bad)
