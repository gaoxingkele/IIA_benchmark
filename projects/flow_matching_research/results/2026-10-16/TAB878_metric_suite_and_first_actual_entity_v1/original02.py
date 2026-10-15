"""Official metric probes,not neural training or benchmark performance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as s


def test_all878_native_cases2057labels87798entity_views_and_all21functions():
    c=s.registration();entries=s.verify(c)
    assert len(entries)==878 and sum(len(e['case']['required_label_policies']) for e in entries)==2057
    assert c['required_entity_label_views']==87798 and len(c['score_metrics'])==6 and len(c['label_metrics'])==15
    assert len({e['key'] for e in entries})==878 and sum(e['track']=='common105_DCD5' for e in entries)==5
    p=s.read(s.ROOT/s.scalar.POLICY)
    assert c['settings']==p['settings'] and c['resource_lock']==p['resource_lock']
    metrics=s.load_metrics(c)
    assert tuple(metrics['score'])==s.SCORE and tuple(metrics['label'])==s.LABEL
    assert all(Path(f.__code__.co_filename).resolve().is_relative_to((s.ROOT/s.TAB).resolve()) for group in metrics.values() for f in group.values())


@pytest.mark.parametrize('fault',['case','metric','labels','memory','lock','thresholds','source','CI'])
def test_narrower_scope_altered_TAB_definition_or_resource_policy_rejected(fault):
    c=s.registration()
    if fault=='case':c['entries'].pop()
    elif fault=='metric':c['score_metrics'].pop()
    elif fault=='labels':c['entries'][0]['case']['required_label_policies'].pop()
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='lock':c['resource_lock']='looser.lock'
    elif fault=='thresholds':c['diagnostic_pooled_ratios']=[1]
    elif fault=='source':c['official_TAB_commit']='current'
    else:c['entity_macro_bootstrap_resamples']=2
    with pytest.raises(ValueError):s.verify(c)


def results(c):
    return dict(metrics=dict(strict={str(r):dict(calibration=dict(threshold=.4+r*.001)) for r in c['strict_ratios']},
        tab_core_diagnostic=dict(ratio_grid=[dict(ratio_percent=r,threshold=.5+r*.001) for r in c['diagnostic_pooled_ratios']])))


def test_all_original_functions_execute_with_private_float64_args_and_all13_thresholds():
    c=s.registration();metrics=s.load_metrics(c);score=np.linspace(0,1,64)
    a=np.zeros(64,np.uint8);a[10:12]=1;a[45:47]=1;b=a.copy();b[12]=1
    old_score,old_a,old_b=score.copy(),a.copy(),b.copy()
    rows=s.evaluate_entity(c,score,dict(first=a,last=b),results(c),metrics)
    s.validate_cells(c,rows,['first','last'])
    assert len(rows['first'])==201 and len(rows['last'])==201
    assert all(cell['status'] in ['finite','nonfinite','error'] for cells in rows.values() for cell in cells.values())
    assert np.array_equal(score,old_score) and np.array_equal(a,old_a) and np.array_equal(b,old_b)
    assert rows['first']['score/auc_roc']['value']==metrics['score']['auc_roc'](a,score)
    assert rows['first']['score/VUS_PR']['value']==metrics['score']['VUS_PR'](a,score)
    direct=metrics['label']['affiliation_f'](a,score>results(c)['metrics']['strict']['1']['calibration']['threshold'])
    assert rows['first']['validation/1/affiliation_f']['value']==direct


def test_original_VUS_uses_unclipped_label_median_all_integer_windows_and250_thresholds():
    c=s.registration();metrics=s.load_metrics(c)
    module=sys.modules['ts_benchmark.evaluation.metrics.classification_metrics_score']
    x=np.linspace(0,1,40);y=np.zeros(40);y[10:12]=1;y[30:34]=1
    sliding=int(np.median(module.get_list_anomaly(y)))
    curve=module.generate_curve(y,x,2*sliding)
    assert metrics['score']['VUS_ROC'](y,x)==curve[-2]
    assert metrics['score']['VUS_PR'](y,x)==curve[-1]
    assert set(curve[1])==set(range(2*sliding+1))
    assert len(curve[1])==(2*sliding+1)*252


@pytest.mark.parametrize('kind',['nonfinite','error','mutation'])
def test_undefined_and_source_exceptions_retained_without_modifying_real_arrays(kind):
    a=np.array([0,1,0]);p=np.array([.1,.9,.2]);original_a=a.copy();original_p=p.copy()
    def function(actual,predicted):
        assert actual.dtype==predicted.dtype==np.float64
        if kind=='error':raise ValueError('preserve original metric exception')
        if kind=='mutation':actual[:]=0;predicted[:]=1;return .25
        return float('nan')
    result=s.evaluate_cell(function,a,p)
    assert np.array_equal(a,original_a) and np.array_equal(p,original_p)
    assert result['status']==dict(nonfinite='nonfinite',error='error',mutation='finite')[kind]
    if kind!='mutation':assert result['value'] is None
    if kind=='error':assert 'preserve original metric exception' in result['error']


@pytest.mark.parametrize('fault',['metric','policy','unknown_status','impute','false_finite'])
def test_missing_or_imputed_cell_cannot_be_certified(fault):
    c=s.registration();names={'score/'+n for n in c['score_metrics']}
    levels=['validation/'+str(r) for r in c['strict_ratios']]+['diagnostic_pooled_TEST/'+str(r) for r in c['diagnostic_pooled_ratios']]
    names.update(level+'/'+n for level in levels for n in c['label_metrics'])
    rows={'first':{n:dict(status='finite',value=.2,error=None) for n in names}}
    cell=next(iter(rows['first'].values()))
    if fault=='metric':rows['first'].pop(next(iter(names)))
    elif fault=='policy':rows['last']=deepcopy(rows['first'])
    elif fault=='unknown_status':cell['status']='skipped'
    elif fault=='impute':cell.update(status='error',value=0,error='failure')
    else:cell['value']=float('nan')
    with pytest.raises(ValueError):s.validate_cells(c,rows,['first'])


def test_missing_entities_not_silently_dropped_and_complete_macro_CI_uses_entity_groups():
    c=s.registration()
    def fragment(v,status='finite'):
        return dict(metric_results=dict(first={'score/auc_roc':dict(status=status,value=v,error=None)}))
    full=s.aggregate(c,[fragment(.2),fragment(.8)],1103)['first']['score/auc_roc']
    assert full['total_entities']==full['finite_entities']==2 and full['complete_entity_macro_mean']==.5
    assert full['complete_entity_macro_CI95']==[.2,.8] and full['CI_resamples']==200
    assert not full['training_seed_uncertainty']
    missing=s.aggregate(c,[fragment(.2),fragment(None,'nonfinite')],1103)['first']['score/auc_roc']
    assert missing['complete_entity_macro_mean'] is None and missing['complete_entity_macro_CI95'] is None
    assert missing['defined_subset_macro_mean']==.2 and missing['finite_entities']==1 and missing['total_entities']==2


def test_existing_complete_native_source_is_ready_and_pending_model_is_not():
    c=s.registration();closed=next(e for e in c['entries'] if e['case']['family']=='tranad64' and e['case']['dataset']=='smap' and e['case']['seed']==1103)
    proof=s.ready_case(c,closed)
    assert proof is not None and len(proof['dataset']['entities'])==54
    assert proof['lineage']['source_binding']['checkpoint_sha256']==proof['result']['checkpoint_sha256']
    pending=next(e for e in c['entries'] if e['track']=='common105_DCD5')
    if s.source_status(pending) is None:assert s.ready_case(c,pending) is None


def test_registration_and_actual_native_readiness_do_not_load_torch_or_numpy():
    code='from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as s;import sys;c=s.registration();s.verify(c);e=next(e for e in c["entries"] if e["case"]["family"]=="tranad64" and e["case"]["dataset"]=="smap" and e["case"]["seed"]==1103);assert s.ready_case(c,e) is not None;assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
