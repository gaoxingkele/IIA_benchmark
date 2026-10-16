from copy import deepcopy
import math
import numpy as np
import pytest
from scripts.flow_matching import evaluate_tranad_original_SPOT_full30_v1 as s


def test_original_zero_shape_extreme_quantile_uses_negative_log_upper_tail():
    ns=s.metrics_namespace('smap');spot=ns['SPOT'](1e-5)
    spot.n=1000;spot.Nt=20;spot.init_threshold=3.
    value=spot._quantile(0.,2.)
    assert value==pytest.approx(3.-2.*math.log(1000*1e-5/20)) and value>3.


def test_original_initialize_uses_level_not_one_minus_level():
    ns=s.metrics_namespace('smap');spot=ns['SPOT'](1e-5)
    train=np.linspace(0,10,1001);spot.fit(train,np.array([1.,3.]))
    spot._grimshaw=lambda:(0.,1.,0.)
    spot.initialize(level=.98,min_extrema=False,verbose=False)
    assert spot.init_threshold==np.sort(train)[int(.98*len(train))]
    assert spot.init_threshold>np.quantile(train,.02)


def test_exact_original_POT_output_and_all_labels_keep_shared_threshold_and_inputs():
    rng=np.random.default_rng(42);train=rng.exponential(1,2000);test=rng.exponential(1,101)
    labels=np.zeros(101,dtype=np.uint8);labels[10:15]=1;other=labels.copy();other[15]=1
    ns=s.metrics_namespace('smap');before=(train.tobytes(),test.tobytes(),labels.tobytes(),other.tobytes())
    np.random.seed(48);expected,pred=ns['pot_eval'](train.copy(),test.copy(),labels.copy(),q=1e-5)
    result,arrays=s.exact_entity(train,test,dict(first=labels,other=other),'first',ns,48,1e-5)
    assert result['threshold']==expected['threshold'] and arrays['first__source_PA'].tobytes()==pred.tobytes()
    assert before==(train.tobytes(),test.tobytes(),labels.tobytes(),other.tobytes())
    assert not result['threshold_uses_TEST_labels']


def test_constant_training_population_is_undefined_not_a_percentile_replacement():
    y=np.array([0,1,0],dtype=np.uint8)
    out,arrays=s.exact_entity(np.ones(100),np.array([0.,1.,3.]),dict(primary=y),'primary',{},1,1e-5)
    assert out['threshold'] is None and out['calibration_undefined_not_percentile_imputed'] and arrays=={}


def test_original_PA_index0_quirk_and_epsilon_are_preserved():
    ns=s.metrics_namespace('smap');score=np.array([0.,0.,2.,0.,0.]);y=np.array([1,1,1,0,0])
    pred=ns['adjust_predicts'](score,y,threshold=1.)
    assert pred.tolist()==[False,True,True,False,False]
    assert ns['calc_point2point'](np.array([1,0]),np.array([1,0]))[0]<1.


def test_nonfinite_source_metric_is_retained_as_explicit_null():
    assert s.outcomes(dict(f1=.3,auc=float('nan')))==dict(values=dict(f1=.3,auc=None),nonfinite_fields=['auc'])


def test_all25_unique_cases_all30_comparison_slots_and_own_source_receipts_remain():
    c=s.registration();assert len(c['entries'])==25 and c['required_comparison_slots']==30
    assert {r['entry']['case']['dataset'] for r in c['entries']}=={'smap','msl','smd'}
    assert len(c['shared_original_SMD_comparison_references'])==5
    assert {r['seed'] for r in c['shared_original_SMD_comparison_references']}==set(range(1103,1108))
    assert all(not r['new_independent_case'] for r in c['shared_original_SMD_comparison_references'])
    assert len({r['entry']['case']['id'] for r in c['entries']})==25
    assert c['original_risk']==1e-5 and c['settings']['minimum_commit_headroom_bytes']==20*2**30
    c['entries'].pop()
    with pytest.raises(ValueError):s.verify(c)


def test_TEST_labels_do_not_change_original_source_calibration():
    rng=np.random.default_rng(51);train=rng.exponential(1,2000);test=rng.exponential(1,101)
    a=np.zeros(101,dtype=np.uint8);a[5:10]=1;b=np.zeros_like(a);b[50:70]=1
    ns=s.metrics_namespace('smap')
    one,_=s.exact_entity(train,test,dict(primary=a),'primary',ns,53,1e-5)
    two,_=s.exact_entity(train,test,dict(primary=b),'primary',ns,53,1e-5)
    assert one['threshold']==two['threshold']
