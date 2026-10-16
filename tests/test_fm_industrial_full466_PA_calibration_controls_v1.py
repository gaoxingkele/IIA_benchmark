"""Real controls are separate from these small protocol-rejection fixtures."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import audit_industrial_full466_PA_calibration_controls_v1 as s
from scripts.flow_matching.timegan_native_common_v1 import read,write,sha


def test_PA_never_bridges_adjacent_different_original_entities():
    r=s.paired_counts([([0.,0.],[1,1]),([0.,2.],[1,1])],1.)
    assert r['no_PA']['true_positives']==1 and r['with_PA']['true_positives']==2
    assert r['ground_truth_events']==2 and r['detected_events']==1


def test_complete_truth_event_credit_preserves_all_false_alarms_and_strict_ties():
    r=s.paired_counts([([1.,0.,2.,1.,3.],[1,1,1,0,0])],1.)
    assert r['no_PA']['true_positives']==1 and r['with_PA']['true_positives']==3
    assert r['no_PA']['false_positives']==r['with_PA']['false_positives']==1
    assert r['no_PA']['false_negatives']==2 and r['with_PA']['false_negatives']==0


@pytest.mark.parametrize('labels,scores',[
    ([1,1,0,1,1],[0.,3.,2.,0.,4.]),([0,0,0],[1.,0.,3.]),([1,1,1],[0.,0.,0.]),
    ([1,1,1],[2.,0.,0.]),([0,1,1],[0.,0.,3.]),([1]*1000,[0.]*999+[3.])])
def test_independent_run_credit_matches_full_source_PA_without_integer_overflow(labels,scores):
    from iia_benchmark.evaluation.mtsad_metrics import point_adjust,pointwise_metrics
    from scripts.flow_matching.audit_crossad_full139_independent_metrics_v1 import equal_fields
    labels=np.asarray(labels,dtype=np.uint8);scores=np.asarray(scores,dtype=float)
    r=s.paired_counts([(list(scores),list(labels))],1.)
    equal_fields(r['no_PA'],pointwise_metrics(labels,scores>1),dict(rtol=1e-13,atol=1e-13))
    equal_fields(r['with_PA'],pointwise_metrics(labels,point_adjust(labels,scores>1)),dict(rtol=1e-13,atol=1e-13))


@pytest.mark.parametrize('entities,threshold',[
    ([],1.),([([0.],[0,1])],1.),([([float('nan')],[0])],1.),([([1.],[2])],1.),([([0.],[0])],float('inf'))])
def test_full_finite_aligned_vectors_and_binary_truth_are_required(entities,threshold):
    with pytest.raises(ValueError):s.paired_counts(entities,threshold)


def fixture(tmp_path,monkeypatch):
    # Source fixture gives all ten original diagnostic ratios, not a model run.
    from tests.test_fm_industrial_full550_independent_metrics_v1 import fixture as old_fixture
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    c,q,m,job,inputs,payload=old_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(s,'ROOT',tmp_path)
    q['evaluation']['tab_threshold_ratios_percent']=s.RATIOS
    source_path=tmp_path/'output/result.json';source=read(source_path)
    dataset=m['datasets'][0]
    entities={g['entity_id']:(payload[g['array_prefix']+'_labels'],payload[g['array_prefix']+'_test']) for g in dataset['entities']}
    source['metrics']=evaluate_saved(entities,payload['v_validation'],payload['t_train'],dataset['input_sha256'],source['checkpoint_sha256'],q['evaluation'],1103)
    write(source_path,source)
    old=s.saved.audit(c,q,m,job);proof=tmp_path/'previous.json';write(proof,old)
    bindings=dict(fixture=dict(actual_execution_id='fixture',source_result_path='output/result.json',source_result_sha256=sha(source_path),
        scores_sha256=source['scores_sha256'],separate_recovery=False,independent_report_path=str(proof),independent_report_sha256=sha(proof)))
    c=dict(c,strict_validation_alarm_percent=[.1,1,5])
    return c,bindings,q,m,source_path


def test_full_ten_source_grid_and_new_strict_same_threshold_PA_controls(tmp_path,monkeypatch):
    c,b,q,m,p=fixture(tmp_path,monkeypatch);r=s.audit(c,b,q,m,'fixture')
    assert r['passed'] and len(r['strict_validation_controls'])==3 and len(r['all_ten_test_pool_diagnostic_controls'])==10
    assert r['test_label_selected_best_diagnostic']['test_labels_used_for_selection']
    assert not r['test_label_selected_best_diagnostic']['primary_performance']
    assert all(not v['threshold_uses_test_scores'] for v in r['strict_validation_controls'].values())
    assert all(v['threshold_uses_test_scores'] for v in r['all_ten_test_pool_diagnostic_controls'])
    assert r['new_neural_forwards']==r['new_optimizer_updates']==0


@pytest.mark.parametrize('fault',['short_grid','PA_value','threshold','best_grid','source_hash'])
def test_frozen_source_grid_and_exact_score_binding_must_reproduce(tmp_path,monkeypatch,fault):
    c,b,q,m,p=fixture(tmp_path,monkeypatch);source=read(p)
    if fault=='source_hash':b['fixture']['source_result_sha256']='wrong'
    else:
        grid=source['metrics']['tab_core_diagnostic']['ratio_grid']
        if fault=='short_grid':grid.pop()
        elif fault=='PA_value':grid[0]['adjust_f_score']+=.1
        elif fault=='threshold':grid[0]['threshold']+=.1
        else:source['metrics']['tab_core_diagnostic']['best_grid_adjust_f_score']+=.1
        write(p,source);b['fixture']['source_result_sha256']=sha(p)
    with pytest.raises(ValueError):s.audit(c,b,q,m,'fixture')


def test_cold_registration_keeps_complete_original_scope_without_neural_imports():
    code=('import sys;from scripts.flow_matching.audit_industrial_full466_PA_calibration_controls_v1 import registration,verify;'
        'c=registration();b,q,m=verify(c);assert len(b)==466 and len(q["jobs"])==550;'
        'assert c["required_full_original_GPU_scope"]==850;'
        'assert c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert c["test_label_selected_best_is_diagnostic_only"];'
        'assert not any(n in sys.modules for n in ["numpy","torch","scipy","sklearn","pandas"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
