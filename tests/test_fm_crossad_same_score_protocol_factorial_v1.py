import copy
import pytest

from scripts.flow_matching import audit_crossad_same_score_protocol_factorial_v1 as s


def test_PA_only_expands_hit_true_events_and_preserves_normal_false_alarms():
    truth = [0,1,1,0,1,1,0]
    pred = [1,1,0,0,0,0,1]
    adjusted = s.point_adjust(truth,pred)
    assert adjusted == [1,True,True,0,0,0,1]
    assert s.confusion(truth,adjusted)['false_positives'] == s.confusion(truth,pred)['false_positives'] == 2
    assert s.confusion(truth,adjusted)['true_positives'] == 2
    with pytest.raises(ValueError):s.point_adjust([0,2],[0,1])


def test_full_factorial_isolates_calibration_and_PA_without_picking_by_test_labels():
    train = [0.,0.,0.,0.]
    val = [0.,1.,2.,3.]
    scores = [0.,0.,4.,0.,1.,2.,0.]
    labels = [0,1,1,0,0,0,0]
    result = s.factorial(train,val,scores,labels,[50.])['50']
    p = result['protocols']
    assert set(p) == set(s.PROTOCOLS)
    assert p['validation_raw']['threshold'] == p['validation_PA']['threshold'] == 1.5
    assert p['pooled_train_test_raw']['threshold'] == p['pooled_train_test_PA']['threshold']
    assert p['validation_PA']['f1'] > p['validation_raw']['f1']
    assert all(not x['test_labels_used_to_select_threshold'] for x in p.values())
    assert not p['validation_raw']['test_scores_used_in_calibration']
    assert p['pooled_train_test_raw']['test_scores_used_in_calibration']
    assert not p['validation_raw']['test_labels_used_in_postprocessing']
    assert p['validation_PA']['test_labels_used_in_postprocessing']
    delta = result['paired_F1_differences']
    assert delta['combined'] == pytest.approx(delta['PA_at_validation']+delta['pooled_calibration_without_PA']+delta['interaction'])


def test_zero_positive_labels_and_zero_alarm_predictions_are_finite():
    assert s.confusion([0,0],[0,0])['f1'] == 0.
    assert s.point_adjust([0,0],[1,0]) == [1,0]
    assert s.point_adjust([1,1],[0,0]) == [0,0]


def test_partial_training_seed_groups_never_get_mean_or_paired_interval():
    q = s.read(s.ROOT/s.parent.QUEUE)
    jobs = [j for j in q['jobs'] if j['dataset']=='GECCO' and not j['reconstructed_ablation'] and j['seed'] in s.groups.SEEDS]
    records = {}
    for job in jobs:
        records[job['canonical_CPU_condition_reference']] = dict(raw_sha256='same',complete_points=dict(test=7),
            all_score_label_and_timestamp_payload_sha256=dict(labels='same',source_rows='same'),
            protocol_factorial={ratio:dict(protocols={p:dict(f1=.5) for p in s.PROTOCOLS},
                paired_F1_differences={k:0. for k in s.DELTAS}) for ratio in ['0.1','1','5']})
    key = jobs[-1]['canonical_CPU_condition_reference']
    missing = records.pop(key)
    summary = s.cohort_summary(q,records)
    assert summary['complete_primary_cohorts'] == 0
    assert 'by_alarm_budget' not in summary['cohorts']['GECCO/main']
    records[key] = missing
    assert s.cohort_summary(q,records)['complete_primary_cohorts'] == 1
    bad = copy.deepcopy(records)
    bad[key]['all_score_label_and_timestamp_payload_sha256']['labels'] = 'changed'
    with pytest.raises(ValueError):s.cohort_summary(q,bad)
