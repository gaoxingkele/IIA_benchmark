"""Source-shaped plumbing checks;these fixtures are not benchmark results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess,sys
import numpy as np
import pytest
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as s


def test_registration_keeps_all873_cases_and_all2037_label_views():
    c=s.registration();cases=s.verify(c)
    assert len(cases)==873 and len({r['id'] for r in cases})==873
    assert sum(len(r['required_label_policies']) for r in cases)==2037
    assert c['required_view_counts']==dict(smap=1164,msl=582,smd=291)
    assert len({r['family'] for r in cases})==7
    p=s.read(s.ROOT/s.POLICY)
    assert c['settings']==p['settings'] and c['resource_lock']==p['resource_lock']
    assert not c['test_based_label_policy_selection_allowed'] and not c['independent_model_reload_or_reinference_performed']
    assert c['original_scope']==dict(papers=45,method_baseline_ablation_records=681,data_tasks=273,paper_scalar_targets=702)


@pytest.mark.parametrize('field',['cases','required_label_view_results','required_view_counts','settings','resource_lock','test_based_label_policy_selection_allowed'])
def test_no_case_policy_or_resource_shortcut_can_enter_registration(field):
    c=s.registration()
    if field=='cases':c[field].pop()
    elif field=='required_label_view_results':c[field]=873
    elif field=='required_view_counts':c[field]['smap']=291
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=1
    elif field=='resource_lock':c[field]='another_lane.lock'
    else:c[field]=True
    with pytest.raises(ValueError):s.verify(c)


def guarded_source(family):
    job=dict(id='fixture_only',dataset='smap',seed=1103,window=100,model_config='fixture.json',output_directory='fixture_unused')
    case=dict(id=job['id'],dataset='smap',seed=1103,window=100,family=family,queue_path='F:/fixture_only/q.json',queue_sha256='fixture_digest')
    gpu=family=='neural'
    guard=dict(exit_code=0,resource_guard_aborted=False,queue_sha256=case['queue_sha256'],
        command=['fixture_python','-m','source','--queue',case['queue_path'],'--job-id',job['id']])
    guard['original_job' if gpu else 'job_id']=job['id']
    guard['full_original_budget_retained' if gpu else 'full_registered_budget_retained']=True
    result=dict(status='completed',id=job['id'],job=deepcopy(job),dataset='smap',seed=1103,model_config='fixture.json',
        purpose='full_frozen_pretrained_zero_shot_tsad' if family=='moment' else 'full_local_tsad_benchmark',
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest())
    return case,job,result,guard


@pytest.mark.parametrize('family',['objectives','neural','baselines','entropic','reflow','tranad64','moment'])
def test_exact_source_guard_and_result_required_for_every_family(family):
    case,job,result,guard=guarded_source(family);s.validate_source_result(case,job,result,guard)
    assert not ('independent_native_replay' in result)


@pytest.mark.parametrize('fault',['failed','resource_abort','budget','source_identity','queue_hash','command_job','command_queue','fingerprint','result_job','purpose','model_config'])
def test_wrong_or_failed_source_is_not_usable_for_metrics(fault):
    case,job,result,guard=guarded_source('baselines')
    if fault=='failed':guard['exit_code']=1
    elif fault=='resource_abort':guard['resource_guard_aborted']=True
    elif fault=='budget':guard['full_registered_budget_retained']=False
    elif fault=='source_identity':guard['job_id']='another_case'
    elif fault=='queue_hash':guard['queue_sha256']='changed'
    elif fault=='command_job':guard['command'][-1]='other'
    elif fault=='command_queue':guard['command'][4]='F:/fixture_only/other.json'
    elif fault=='fingerprint':result['experiment_sha256']='wrong'
    elif fault=='result_job':result['job']['seed']=42
    elif fault=='purpose':result['purpose']='smoke_only'
    else:result['model_config']='another.json'
    with pytest.raises(ValueError):s.validate_source_result(case,job,result,guard)


def test_only_correct_source_failure_is_recorded_as_a_negative_outcome(tmp_path,monkeypatch):
    case,job,result,guard=guarded_source('entropic');guard['exit_code']=1
    path=tmp_path/'receipt.json';path.write_text(json.dumps(guard));case['source_guard']=str(path)
    monkeypatch.setattr(s,'resolve_case',lambda case:({},job))
    outcome=s.source_status(case)
    assert outcome['kind']=='source_failed_preserved' and not outcome['metrics_available']
    guard['job_id']='not_the_original_case';path.write_text(json.dumps(guard))
    with pytest.raises(ValueError):s.source_status(case)


def test_source_result_alone_is_insufficient_before_guarded_worker_exits(tmp_path,monkeypatch):
    case,job,result,guard=guarded_source('moment');job['output_directory']=str(tmp_path)
    (tmp_path/'result.json').write_text(json.dumps(result));case['source_guard']=str(tmp_path/'missing_guard.json')
    monkeypatch.setattr(s,'resolve_case',lambda case:({},job))
    assert s.source_status(case) is None


class Arrays(dict):
    @property
    def files(self):return list(self)


def fixture_arrays():
    dataset=dict(input_sha256='1'*64,primary_label_policy='first_exclusive',labels_only_controls={p:{} for p in ['first_exclusive','first_inclusive','last_exclusive','last_inclusive']},
        entities=[dict(entity_id='fixture_A',array_prefix='e000'),dict(entity_id='fixture_B',array_prefix='e001')])
    saved=Arrays();inputs=Arrays()
    for i in range(2):
        prefix=f'e{i:03d}'
        for role,values in [('train',[.1,.2,.1,.3]),('validation',[.1,.5]),('test',[.1,.7,.4,.2] if i==0 else [.4,.3,.2,.9])]:
            saved[prefix+'_'+role]=np.asarray(values,dtype=np.float64);inputs[prefix+'_'+role]=np.ones((len(values),2),dtype=np.float32)
        saved[prefix+'_labels']=np.array([0,1,1,0] if i==0 else [0,0,0,1],dtype=np.uint8)
        inputs[prefix+'_labels']=saved[prefix+'_labels'].copy()
    return inputs,saved,dataset


@pytest.mark.parametrize('fault',[None,'omitted_point','NaN','wrong_labels','extra_array'])
def test_every_saved_point_label_and_array_requires_audit(fault):
    inputs,saved,dataset=fixture_arrays()
    if fault=='omitted_point':saved['e000_test']=saved['e000_test'][:-1]
    elif fault=='NaN':saved['e001_train'][0]=np.nan
    elif fault=='wrong_labels':saved['e000_labels'][0]=1
    elif fault=='extra_array':saved['untracked']=np.array([1])
    if fault:
        with pytest.raises(ValueError):s.collect_arrays(inputs,saved,dataset)
    else:
        entities,val,train,coverage,digests=s.collect_arrays(inputs,saved,dataset)
        assert len(entities)==2 and len(coverage)==6 and len(digests)==8 and len(val)==4 and len(train)==8


def views_fixture():
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    inputs,saved,dataset=fixture_arrays();entities,validation,training,_,_=s.collect_arrays(inputs,saved,dataset)
    # Reduced resampling is exclusive to this tiny algorithmic unit fixture.
    evaluation=dict(strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,bootstrap_resamples=8,
        bootstrap_block_points=2,tab_threshold_ratios_percent=[.5,1,2,5,10],tab_commit='fixture_only')
    result=dict(checkpoint_sha256='2'*64,scores_sha256='3'*64)
    result['metrics']=evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],evaluation,1103)
    overlays={p:{e['array_prefix']+'_labels':entities[e['entity_id']][0].copy() for e in dataset['entities']} for p in dataset['labels_only_controls']}
    overlays['first_inclusive']['e000_labels'][3]=1
    overlays['last_exclusive']['e000_labels']=np.array([0,0,1,1],dtype=np.uint8)
    overlays['last_inclusive']['e000_labels']=np.array([1,0,1,1],dtype=np.uint8)
    return entities,validation,training,overlays,dataset,result,evaluation


def test_all_views_preserve_predictions_validation_calibration_and_every_metric():
    entities,val,train,overlays,dataset,result,evaluation=views_fixture()
    before=deepcopy((entities,overlays));views=s.evaluate_views(entities,val,train,overlays,dataset,result,evaluation,1103)
    assert len(views)==4 and views['first_exclusive']['metrics']==result['metrics']
    assert views['first_inclusive']['changed_target_points']==1
    for name,view in views.items():
        assert not view['label_policy_selected_by_TEST_performance'] and view['source_scores_sha256']=='3'*64
        assert len(view['metrics']['strict'])==3 and len(view['metrics']['tab_core_diagnostic']['ratio_grid'])==5
        assert set(view['independent_confusion_ranking_PA_and_all_grid_controls'])==set(result['metrics']['strict'])
        for ratio,record in view['metrics']['strict'].items():
            assert record['calibration']==result['metrics']['strict'][ratio]['calibration']
            assert record['f1_ci95']['resamples']==8 and record['f1_ci95']['seed']==11103
        for key,array in overlays[name].items():assert np.array_equal(array,before[1][name][key])
    for key,(_,scores) in entities.items():assert np.array_equal(scores,before[0][key][1])


@pytest.mark.parametrize('fault',['missing_policy','missing_entity','truncated_labels','nonbinary','primary_label_mutation','source_metric_mutation'])
def test_partial_or_selected_label_views_and_changed_source_metrics_fail(fault):
    entities,val,train,overlays,dataset,result,evaluation=views_fixture()
    if fault=='missing_policy':overlays.pop('last_inclusive')
    elif fault=='missing_entity':overlays['last_inclusive'].pop('e001_labels')
    elif fault=='truncated_labels':overlays['last_inclusive']['e000_labels']=np.array([0,1])
    elif fault=='nonbinary':overlays['last_inclusive']['e000_labels'][0]=2
    elif fault=='primary_label_mutation':overlays['first_exclusive']['e000_labels'][0]=1
    else:result['metrics']['strict']['1']['micro']['f1']=.99
    with pytest.raises(ValueError):s.evaluate_views(entities,val,train,overlays,dataset,result,evaluation,1103)


def test_cold_full_registration_does_not_import_neural_or_array_runtime():
    code='from scripts.flow_matching import audit_original_full873_label_metrics_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_registration_is_identical_in_author_Python310_and_controller_Python312():
    original=s.registration()
    native=next(c['worker_python'] for c in original['cases'] if c['family']=='moment')
    code='from scripts.flow_matching import audit_original_full873_label_metrics_v1 as s;import json,sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules;print(json.dumps(c,sort_keys=True))'
    result=subprocess.run([str(s.ROOT/native),'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)==original
