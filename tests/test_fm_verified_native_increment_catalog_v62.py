from copy import deepcopy
import pytest
from scripts.flow_matching import publish_verified_native_increment_catalog_v62 as s


def rows():
    fields=['algorithm','dataset','protocol','metric','pattern','missing_ratio','score_recipe','n','required_repeats','mean',
        'task','std','se','ci95_low','ci95_high','captured_utc','author_equivalence_certified','strict_TAB_result','evaluator_repeats']
    pending=dict.fromkeys(fields,'');pending.update(algorithm='model',dataset='data',protocol='frozen_case',metric='F1',
        n='0',required_repeats='1',task='anomaly_detection',captured_utc='original_time',author_equivalence_certified='False',strict_TAB_result='False')
    complete=dict(pending,n='1',mean='.4',captured_utc='actual_completion_time')
    legacy=dict(pending,algorithm='legacy',n='1',mean='.2')
    return [legacy,pending],[complete]


def test_changes_only_actual_pending_control_and_preserves_legacy_capture_time():
    before,update=rows();after=s.merge_pending(before,update)
    assert after[0]==before[0] and after[1]==update[0] and before[1]['n']=='0'


@pytest.mark.parametrize('fault',['already_complete_undefined','already_complete_finite','duplicate_update','missing_row','seed_CI','false_author_cert','missing_evaluator_repeat'])
def test_refuses_overwrite_undefined_finished_case_fabricated_CI_or_incomplete_scope(fault):
    before,update=rows()
    if fault=='already_complete_undefined':before[1]['n']='1'
    if fault=='already_complete_finite':before[1].update(n='1',mean='.1')
    if fault=='duplicate_update':update.append(dict(update[0]))
    if fault=='missing_row':update[0]['protocol']='not_registered'
    if fault=='seed_CI':update[0]['ci95_low']='.1'
    if fault=='false_author_cert':update[0]['author_equivalence_certified']='True'
    if fault=='missing_evaluator_repeat':update[0].update(task='generation',evaluator_repeats='4')
    with pytest.raises(ValueError):s.merge_pending(before,update)
