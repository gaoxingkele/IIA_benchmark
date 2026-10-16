import itertools
import math
import pytest
from scripts.flow_matching import audit_crossad_full_test_label_oracle_v1 as s


def brute(scores,labels,pa):
    best=0.
    thresholds=[max(scores)]+[math.nextafter(x,-math.inf) for x in sorted(set(scores),reverse=True)]
    for t in thresholds:
        prediction=[v>t for v in scores]
        if pa:prediction=s.source.point_adjust(labels,prediction)
        best=max(best,s.source.confusion(labels,prediction)['f1'])
    return best


@pytest.mark.parametrize('scores',[[0.,0.,1.,1.],[0.,3.,1.,2.],[-2.,-1.,-1.,0.]])
def test_exhaustive_tied_score_sweep_matches_every_binary_truth_and_both_full_protocols(scores):
    for labels in itertools.product([0,1],repeat=4):
        result=s.oracle(scores,labels)
        assert result['raw']['f1']==brute(scores,labels,False)
        assert result['PA']['f1']==brute(scores,labels,True)
        for row in result.values():
            assert row['test_labels_used_to_select_threshold'] and row['diagnostic_only']
            assert row['evaluated_full_test_points']==4 and not row['deployable_validation_only_result']


def test_true_event_max_transform_equals_full_PA_at_every_threshold_including_normal_alarms():
    labels=[0,1,1,0,1,1,1,0];scores=[2.,0.,3.,4.,1.,1.,0.,2.]
    transformed=s.pa_scores(scores,labels)
    for threshold in [-math.inf,0.,1.,2.,3.,4.,math.inf]:
        expected=s.source.point_adjust(labels,[x>threshold for x in scores])
        assert [x>threshold for x in transformed]==expected


def test_oracle_handles_zero_positives_and_uses_most_conservative_exact_tie():
    row=s.exact_best_f1([0.,1.,1.],[0,0,0])
    assert row['f1']==0. and row['threshold']==1. and row['false_positives']==0
    with pytest.raises(ValueError):s.exact_best_f1([math.nan],[1])
    with pytest.raises(ValueError):s.exact_best_f1([1.,2.],[1])
    with pytest.raises(ValueError):s.pa_scores([1.],[2])


def test_registration_labels_oracle_threshold_optimization_and_fixed_controls_separately():
    c=s.registration()
    assert c['test_label_threshold_selection']
    assert not c['original_fixed_calibration_controls_test_label_threshold_selection']
    assert c['oracle_scores_never_enter_strict_leaderboards']
    assert c['required_conditions']==139 and c['required_primary_cohorts']==22
