"""Protocol contrasts must retain support,evidence strength and diagnostic labels."""
from copy import deepcopy

import pytest

from scripts.flow_matching import summarize_verified_protocol_effects_v1 as subject


def test_PA_only_contrast_keeps_truth_and_false_positive_decisions_fixed():
    before=dict(tp=1,fp=1,fn=9,tn=89);after=dict(tp=10,fp=1,fn=0,tn=89)
    result=subject.fixed_threshold_effect(before,after,100,2/12,20/21)
    assert result['newly_credited_anomaly_points']==9
    assert result['F1_increase_percentage_points']==pytest.approx(100*(20/21-2/12))
    assert not result['model_changed'] and result['fixed_test_support']


@pytest.mark.parametrize('after',[dict(tp=10,fp=2,fn=0,tn=88),dict(tp=9,fp=1,fn=0,tn=90),dict(tp=.5,fp=1,fn=9.5,tn=89)])
def test_cannot_label_changed_normal_predictions_truth_or_fractional_counts_as_PA_only(after):
    with pytest.raises(ValueError):subject.fixed_threshold_effect(dict(tp=1,fp=1,fn=9,tn=89),after,100,2/12,.9)


def test_actual_closed_cases_keep_saved_score_and_native_evidence_distinct():
    report=subject.build_report(subject.configuration())
    assert report['case_count']==4 and report['method_families']==3 and report['datasets']==3
    source,causal,moment,sb=report['cases']
    assert source['native_model_reinference_proven'] and causal['native_model_reinference_proven']
    assert source['same_threshold_effect']['F1_increase_percentage_points']>95
    assert source['same_threshold_effect']['no_PA_confusion']['fp']==source['same_threshold_effect']['PA_confusion']['fp']
    assert moment['independent_saved_stream_and_metric_review_proven'] and not moment['native_model_reinference_proven']
    assert not source['original_training_guard_proven']
    assert sb['primary_test_normal_false_alarm_fraction']>.5
    assert not report['new_seed_aggregate_confidence_intervals_or_pvalues']
    assert report['comparisons_are_within_case_not_cross_model_ranking']


def test_test_grid_selection_effect_is_not_mislabeled_as_PA_only_or_model_improvement():
    report=subject.build_report(subject.configuration());sb=report['cases'][-1]
    assert sb['same_threshold_effect']['F1_increase_percentage_points']>50
    assert sb['best_test_label_selected_plain_f1']>sb['primary_validation_only_no_PA_f1']
    assert report['sb_distinct_operations']['latter_is_diagnostic_multiple_protocol_changes_not_isolated_PA']
    assert not sb['model_changed_between_grid_controls']


def test_changed_archival_source_hash_is_rejected():
    config=subject.configuration();config['evidence'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='evidence/code changed'):subject.build_report(config)
