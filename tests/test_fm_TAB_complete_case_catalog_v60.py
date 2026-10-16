from copy import deepcopy
from pathlib import Path
import json
import pytest
from scripts.flow_matching import publish_TAB_complete_case_catalog_v60 as s

PACKET=s.ROOT/'projects/flow_matching_research/results/2026-10-16/TAB878_TranAD64_SMAP1104_full54_v1'


def rows(report=None):
    report=report or s.read(PACKET/'complete_case_TAB_metric_controls.json')
    fields=['algorithm','dataset','task','protocol','metric','n','required_repeats','mean','std','se','ci95_low','ci95_high','paper_mean',
        'author_equivalence_certified','strict_TAB_result','source','captured_utc','comparison','score_recipe','metric_unit','selection_boundary','uncertainty_scope']+list(s.ENTITY_FIELDS)
    return s.complete_case_rows(fields,report,PACKET/'complete_case_TAB_metric_controls.json','frozen')


def test_all804_real_controls_keep700_complete_and104_undefined_macros():
    result=rows();assert len(result)==804 and sum(bool(x['mean']) for x in result)==700
    undefined=[r for r in result if not r['mean']];assert len(undefined)==104
    assert all(r['n']=='1' and int(r['entity_finite'])<54 and not r['entity_bootstrap_ci95'] and r['defined_subset_macro_mean'] for r in undefined)
    assert all(r['author_equivalence_certified']==r['strict_TAB_result']=='False' for r in result)
    assert all(not r[k] for r in result for k in ['std','se','ci95_low','ci95_high','paper_mean'])


@pytest.mark.parametrize('fault',['coverage','imputation','CI','seedCI','drop_policy','drop_metric','equivalence'])
def test_missing_coverage_imputed_full_macro_or_false_certification_is_rejected(fault):
    report=deepcopy(s.read(PACKET/'complete_case_TAB_metric_controls.json'))
    first=next(iter(report['macro_metrics'].values()))
    partial=next(v for v in first.values() if v['finite_entities']<54)
    if fault=='coverage':partial['finite_entities']+=1
    elif fault=='imputation':partial['complete_entity_macro_mean']=partial['defined_subset_macro_mean']
    elif fault=='CI':partial['complete_entity_macro_CI95']=[0,1]
    elif fault=='seedCI':partial['training_seed_uncertainty']=True
    elif fault=='drop_policy':report['macro_metrics'].pop(next(iter(report['macro_metrics'])))
    elif fault=='drop_metric':first.pop(next(iter(first)))
    else:report['full_TAB_training_or_author_harness_equivalence']=True
    with pytest.raises(ValueError):rows(report)


def test_TEST_score_diagnostics_validation_and_PA_boundaries_stay_explicit():
    result=rows()
    diagnostic=next(r for r in result if json.loads(r['score_recipe'])['official_metric_control']=='diagnostic_pooled_TEST/1/adjust_f_score')
    assert 'train+TEST-score' in diagnostic['selection_boundary'] and 'PA uses TEST labels' in diagnostic['selection_boundary']
    strict=next(r for r in result if json.loads(r['score_recipe'])['official_metric_control']=='validation/1/f_score')
    assert 'validation-only' in strict['selection_boundary'] and 'Original PA' not in strict['selection_boundary']
    assert strict['mean']=='0.08235281807605767' and strict['entity_finite']=='54'
