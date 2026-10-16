"""Small fixtures validate diagnostic math only, never benchmark performance."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.flow_matching import audit_common105_validation_calibration_v1 as s


def test_entity_calibration_uses_only_each_own_validation_scores():
    v={'A':np.zeros(4),'B':np.full(4,100.)};r=s.validation_thresholds(v)
    assert r['global_threshold']==100 and r['per_entity_thresholds']=={'A':0.,'B':100.}
    assert s.validation_thresholds(dict(reversed(list(v.items()))))==r
    for k in v:assert np.array_equal(v[k],[0.]*4 if k=='A' else [100.]*4)
    a=np.array([1.,0.]);b=np.array([101.,100.]);y=np.array([1,0])
    global_counts=np.array([s.confusion(y,a>100),s.confusion(y,b>100)])
    entity_counts=np.array([s.confusion(y,a>0),s.confusion(y,b>100)])
    assert s.point_metrics(global_counts.sum(axis=0))['f1']==pytest.approx(2/3)
    assert s.point_metrics(entity_counts.sum(axis=0))['f1']==1
    # Test labels are absent from the calibration interface; predictions stay
    # fixed even when a labels-only sensitivity changes evaluation counts.
    assert s.confusion(1-y,a>0)!=s.confusion(y,a>0)
    assert s.validation_thresholds(v)==r


@pytest.mark.parametrize('v',[{}, {'A':[]},{'A':[[1,2]]},{'A':[1,np.inf]},{'A':[np.nan]}])
def test_missing_or_nonfinite_validation_cannot_be_silently_filtered(v):
    with pytest.raises(ValueError):s.validation_thresholds(v)


def test_exact_paired_entity_uncertainty_uses_same_resampled_groups():
    r=s.paired_bootstrap([[1,0,1,1],[1,0,1,1]],[[2,0,0,1],[2,0,0,1]],1103)
    assert r['resamples']==200 and r['independent_entity_groups']==2
    assert len(r['paired_F1_difference_samples'])==200
    assert r['entity_bootstrap_F1_difference_ci95']==pytest.approx([1/3,1/3])
    assert r['observed_F1_difference']==pytest.approx(1/3)


@pytest.mark.parametrize('b',[[[2,0,0,0]],[[0,0,2,0]],[[1,-1,1,1]]])
def test_unpaired_or_invalid_entity_populations_are_rejected(b):
    with pytest.raises(ValueError):s.paired_bootstrap([[1,0,0,1]],b,1103)


def test_all_original_matched_cases_labels_and_resource_gates_remain_required():
    q=s.registration();old=s.read(s.ROOT/s.upstream.CONFIG)
    assert len(q['cases'])==len({c['id'] for c in q['cases']})==577
    assert sum(len(c['required_label_policies']) for c in q['cases'])==1726
    assert q['cases']==old['cases'] and q['settings']==old['settings'] and q['resource_lock']==old['resource_lock']
    assert q['diagnostic_protocol']['global_and_entity_validation_false_alarm_percent']==1
    assert q['diagnostic_protocol']['thresholds_never_fit_on_test_scores_or_any_test_labels']
    assert q['diagnostic_protocol']['new_posthoc_diagnostic_not_prospective_preregistration']
