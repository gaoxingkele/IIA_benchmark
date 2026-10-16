from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import audit_industrial_full466_native_ranges_v1 as s


def fixture():
    science=dict(source_receipts=[dict(path='native',sha256='frozen')],tab_commit='fixed')
    config=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest()
    cal=lambda r:dict(calibration_split='validation',false_alarm_percent=float(r),
        rule='validation_score_percentile_linear_strict_greater',threshold=0.9)
    source=dict(id='case',dataset='original',model_config='fixed',seed=1103,scores_sha256='scores',
        metrics=dict(strict={r:dict(calibration=cal(r)) for r in ['0.1','1','5']},
            tab_core_diagnostic=dict(ratio_grid=[dict(ratio_percent=float(i+1),threshold=i/10) for i in range(10)])))
    aff=dict(status='defined',affiliation_f=0.5,affiliation_precision=0.5,affiliation_recall=0.5)
    vus=dict(status='defined',VUS_ROC=0.8,VUS_PR=0.7,max_buffer=4,buffers_evaluated=5,threshold_samples=250)
    em=lambda x,ks:s.defined_macro({'e':x},ks)
    grid=source['metrics']['tab_core_diagnostic']['ratio_grid']
    e=dict(status='completed_range_evaluation',id='case',dataset='original',model_config='fixed',seed=1103,
        model_result_sha256='result',scores_sha256='scores',evaluation_config_sha256=config,
        evaluation_identity=hashlib.sha256(('result'+config).encode()).hexdigest(),**science,
        VUS_per_entity={'e':vus},VUS_entity_macro=em(vus,['VUS_ROC','VUS_PR']),
        strict_affiliation_per_entity={'e':{r:deepcopy(aff) for r in ['0.1','1','5']}},
        strict_affiliation_entity_macro={r:em(aff,s.AFFILIATION) for r in ['0.1','1','5']},
        tab_ratio_affiliation_per_entity={'e':[dict(r,**aff) for r in grid]},
        tab_ratio_affiliation_entity_macro=[dict(r,metrics=em(aff,s.AFFILIATION)) for r in grid])
    return e,source,science,{'e':s.geometry([0,1,1,0,0])}


def test_valid_complete_native_source_rules_and_geometry():
    s.validate_ranges(*fixture())


@pytest.mark.parametrize('fault',['seed','score','config','identity','tab','source','entity','buffer','buffer_count',
    'threshold_count','strict_entity','strict_macro','grid_size','grid_threshold','entity_grid_threshold',
    'macro_count','macro_value','undefined_zero','test_calibration','strict_percent','strict_rule','aff_status'])
def test_reject_altered_source_or_reduced_geometry_and_protocol(fault):
    e,src,c,shapes=fixture()
    if fault=='seed':e['seed']=1104
    elif fault=='score':e['scores_sha256']='changed'
    elif fault=='config':e['evaluation_config_sha256']='changed'
    elif fault=='identity':e['evaluation_identity']='changed'
    elif fault=='tab':e['tab_commit']='changed'
    elif fault=='source':e['source_receipts']=[]
    elif fault=='entity':e['VUS_per_entity']['extra']=e['VUS_per_entity']['e']
    elif fault=='buffer':e['VUS_per_entity']['e']['max_buffer']=2
    elif fault=='buffer_count':e['VUS_per_entity']['e']['buffers_evaluated']=3
    elif fault=='threshold_count':e['VUS_per_entity']['e']['threshold_samples']=25
    elif fault=='strict_entity':del e['strict_affiliation_per_entity']['e']['0.1']
    elif fault=='strict_macro':del e['strict_affiliation_entity_macro']['5']
    elif fault=='grid_size':e['tab_ratio_affiliation_entity_macro'].pop()
    elif fault=='grid_threshold':e['tab_ratio_affiliation_entity_macro'][0]['threshold']+=0.1
    elif fault=='entity_grid_threshold':e['tab_ratio_affiliation_per_entity']['e'][0]['threshold']+=0.1
    elif fault=='macro_count':e['VUS_entity_macro']['VUS_ROC']['undefined_entities']=1
    elif fault=='macro_value':e['VUS_entity_macro']['VUS_ROC']['mean_over_defined_entities']=0.9
    elif fault=='undefined_zero':e['strict_affiliation_per_entity']['e']['1'].update(status='undefined_reference_nan')
    elif fault=='test_calibration':src['metrics']['strict']['1']['calibration']['calibration_split']='test'
    elif fault=='strict_percent':src['metrics']['strict']['1']['calibration']['false_alarm_percent']=5
    elif fault=='strict_rule':src['metrics']['strict']['1']['calibration']['rule']='optimized_test_f1'
    else:e['strict_affiliation_per_entity']['e']['1']['status']='unknown'
    with pytest.raises(ValueError):s.validate_ranges(e,src,c,shapes)


@pytest.mark.parametrize('labels,expected',[
    ([0,1,1,0,1,1,1,1,0],6),([1,1,0,0,0,1],2),([0,0,0],None),([1,1,1],None)])
def test_exact_median_truncation_and_full_endpoint_events(labels,expected):
    assert s.geometry(labels)['max_buffer']==expected


@pytest.mark.parametrize('labels',[[],[0,2,1],[0,-1]])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):s.geometry(labels)


def test_undefined_macros_preserve_denominator_and_null():
    records={'a':dict(x=None),'b':dict(x=0.8)}
    actual={'x':dict(defined_entities=1,undefined_entities=1,mean_over_defined_entities=0.8)}
    s.equal_macro(records,actual,['x'])
    with pytest.raises(ValueError):s.equal_macro(records,{'x':dict(defined_entities=2,undefined_entities=0,mean_over_defined_entities=0.4)},['x'])


def test_single_class_never_gets_zero_vus():
    e,src,c,shapes=fixture();shapes['e']=s.geometry([1,1,1,1,1])
    with pytest.raises(ValueError):s.validate_ranges(e,src,c,shapes)


def test_complete_seed_summary_does_not_reduce_to_defined_subset_or_clip_interval():
    result=s.complete_stats([None,0.4,0.5,0.6,0.7])
    assert result['mean'] is None and result['n']==5 and result['defined_seed_count']==4
    result=s.complete_stats([0,0,0,0,1])
    assert result['mean']==0.2 and result['t95_low']<0
    with pytest.raises(ValueError):s.complete_stats([0.4]*4)
    with pytest.raises(ValueError):s.complete_stats([float('nan')]*5)


def test_cold_full_registration_provenance_and_original_scope():
    code=('import sys;from scripts.flow_matching import audit_industrial_full466_native_ranges_v1 as s;'
        'c=s.registration();q,m,j,e,b=s.verify(c);assert len(q["jobs"])==550 and len(b)==466;'
        'assert c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert not c["independent_VUS_reexecution"] and not c["independent_affiliation_integrals"];'
        'assert not any(n in sys.modules for n in ["numpy","torch","pandas","sklearn","scipy"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
