import math
import subprocess
import sys
import pytest
from scripts.flow_matching import diagnose_industrial_full466_score_direction_v1 as s


def test_both_fixed_validation_tails_are_reported_without_test_selection():
    r=s.controls([0.,1.,2.,3.,4.],[('e',[3.,4.,0.,-1.],[0,0,1,1])])
    assert len(r)==6 and r[1,'1']['micro_and_macro']['f1']==0 and r[-1,'1']['micro_and_macro']['f1']==1
    assert r[1,'1']['micro_and_macro']['auroc']==0 and r[-1,'1']['micro_and_macro']['auroc']==1
    assert sum(x['actual_primary_evaluation'] for x in r.values())==1
    assert all(not x['direction_selected_using_test_labels'] and not x['threshold_uses_test_scores'] and not x['point_adjustment'] for x in r.values())
    assert not any(x['opposite_orientation_primary'] for x in r.values())


@pytest.mark.parametrize('labels',[[0,1,0,1],[0,0,0,0],[1,1,1,1]])
def test_ties_and_undefined_rank_metrics_are_preserved(labels):
    r=s.controls([0.,0.,0.],[('e',[0.]*4,labels)])
    for sign in [1,-1]:assert r[sign,'1']['micro_and_macro']['f1']==0
    expected=.5 if 0<sum(labels)<4 else None
    assert r[1,'1']['micro_and_macro']['auroc']==r[-1,'1']['micro_and_macro']['auroc']==expected


def test_flipping_labels_cannot_choose_a_direction_or_change_thresholds():
    a=s.controls([0.,1.,2.,3.],[('e',[0.,1.,2.,3.],[0,0,1,1])])
    b=s.controls([0.,1.,2.,3.],[('e',[0.,1.,2.,3.],[1,1,0,0])])
    for key in a:
        assert a[key]['transformed_score_threshold']==b[key]['transformed_score_threshold']
        assert a[key]['actual_primary_evaluation']==b[key]['actual_primary_evaluation']
        assert a[key]['calibration_score_payload_sha256']==b[key]['calibration_score_payload_sha256']


def test_threshold_ties_remain_strict_and_event_boundaries_do_not_merge():
    r=s.controls([1.,1.],[('left',[1.,2.],[1,1]),('right',[1.,1.],[1,1])])
    assert r[1,'1']['micro_and_macro']['true_positives']==1
    assert r[1,'1']['event_metrics']['events']==2 and r[1,'1']['event_metrics']['detected_events']==1
    assert r[-1,'1']['micro_and_macro']['true_positives']==0


@pytest.mark.parametrize('validation,entities',[
    ([],[('e',[0.],[0])]),([float('nan')],[('e',[0.],[0])]),([0.],[]),
    ([0.],[('e',[0.,1.],[0])]),([0.],[('e',[0.],[2])]),([0.],[('e',[float('inf')],[0])]),
    ([0.],[('e',[0.],[0]),('e',[1.],[1])])])
def test_missing_nonfinite_misaligned_or_duplicate_native_population_rejected(validation,entities):
    with pytest.raises(ValueError):s.controls(validation,entities)


def test_only_full_paired_five_seed_groups_have_uncertainty_and_no_clipping():
    r=s.paired_summary([0.]*5,[0.,0.,0.,0.,1.])
    assert r['n']==5 and r['source_mean']==0 and r['opposite_mean']==.2 and r['paired_difference_t95_low']<0
    with pytest.raises(ValueError):s.paired_summary([0.]*4,[1.]*4)
    assert s.paired_summary([None,0.,0.,0.,0.],[1.]*5)['paired_difference_mean'] is None
    with pytest.raises(ValueError):s.paired_summary([float('nan')]*5,[1.]*5)


def test_cold_registration_keeps_full_native_scope_and_neural_gate_separate():
    code=('import sys;from scripts.flow_matching import diagnose_industrial_full466_score_direction_v1 as s;'
        'c=s.registration();q,m,j,e,b=s.verify(c);assert len(q["jobs"])==550 and len(b)==466;'
        'assert c["directions"]==[1,-1] and c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert s.read(s.ROOT/s.native.CONFIG)["settings"]["minimum_commit_headroom_bytes"]==32*2**30;'
        'assert not c["direction_selected_using_test_labels"] and not c["opposite_orientation_primary"];'
        'assert not any(n in sys.modules for n in ["numpy","torch","scipy","pandas","sklearn"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
