from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.flow_matching import run_objective_full440_statistics_v1 as s
from scripts.flow_matching.register_objective_full440_statistics_v1 import registration


def all_cases():
    result = {}
    for family,(path,_) in s.producer.QUEUES.items():
        q=s.read(s.ROOT/path)
        for job in q['jobs']:
            result[job['id']]=dict(family=family,path=path,queue=q,job=job)
    return result


def example_metrics():
    # Schema fixture only; no training, scores or benchmark performance claim.
    return dict(strict={ratio:dict(protocol='strict_validation_only_no_pa_v1',point_adjustment=False,
        calibration=dict(calibration_split='validation'),micro=dict(precision=.5,recall=.25,f1=1/3,auroc=.6,average_precision=.4),
        macro_f1=1/3,event_metrics=dict(event_recall=.5,events=2,detected_events=1,mean_detection_delay_points=3.,false_alarms_per_1000_normal_points=2.),
        per_entity=dict(fixture=dict(f1=1/3))) for ratio in ['0.1','1','5']},
        tab_core_diagnostic=dict(test_scores_used_in_calibration=True,test_labels_used_to_select_best_grid=True,
            ratio_grid=[dict(ratio_percent=r,f_score=.4,adjust_f_score=.8) for r in [.1,.5,1,2,3,5,10,15,20,25]],
            best_grid_f_score=.4,best_grid_adjust_f_score=.8,auc_roc=.6,auc_pr=.4))


def test_all206_tasks_cover_every_original_case_and_distinct_control():
    cases=all_cases();selected=s.tasks(cases)
    assert {k:sum(t['kind']==k for t in selected.values()) for k in ['cohort','paired','macro']} == dict(cohort=88,paired=107,macro=11)
    assert {i for t in selected.values() if t['kind']=='cohort' for i in t['cases']} == set(cases)
    assert {k:sum(t.get('contrast')==k for t in selected.values()) for k in ['method','preprocessing','score_head']} == dict(method=80,preprocessing=11,score_head=16)
    for t in selected.values():
        assert len(t['cases'])==dict(cohort=5,paired=10,macro=35)[t['kind']]
        if t['kind']=='paired':
            assert [cases[i]['job']['seed'] for i in t['cases'][:5]] == [cases[i]['job']['seed'] for i in t['cases'][5:]] == s.SEEDS
            assert len({cases[i]['job']['dataset'] for i in t['cases']}) == 1


@pytest.mark.parametrize('mutation',['missing','duplicate'])
def test_incomplete_or_duplicate_training_seed_cannot_form_a_cohort(mutation):
    cases=all_cases();first=next(iter(cases))
    if mutation=='missing':cases.pop(first)
    else:cases['duplicate']=cases[first]
    with pytest.raises(ValueError):s.tasks(cases)


def test_student_interval_and_seed_robustness_match_independent_math():
    from scipy.stats import t
    values=[.1,.2,.3,.4,.5];row=s.five(values)
    mean=sum(values)/5; variance=sum((v-mean)**2 for v in values)/4
    assert s.T_CRITICAL==pytest.approx(float(t.ppf(.975,4)),abs=1e-12)
    assert row['sample_standard_deviation']==pytest.approx(math.sqrt(variance))
    assert row['student_t_df4_95_interval']==pytest.approx([mean-float(t.ppf(.975,4))*math.sqrt(variance/5),mean+float(t.ppf(.975,4))*math.sqrt(variance/5)])
    assert row['leave_one_seed_out_means']['1103']==pytest.approx(.35)
    assert s.five([0,0,0,0,1])['student_t_df4_95_interval'][0]<0


@pytest.mark.parametrize('values',[[1,2,3,4],[1,2,3,4,float('nan')],[1,2,3,4,True],[1,2,3,4,None]])
def test_partial_or_nonfinite_values_never_produce_a_five_seed_interval(values):
    with pytest.raises(ValueError):s.five(values)


def test_undefined_endpoints_keep_all_five_slots_without_cherry_picking():
    rows=[dict(f1=i/10,delay=None if i==0 else i) for i in range(5)]
    result=s.summarize(rows)
    assert result['f1']['summary']['n']==5
    assert result['delay']['defined_seeds']==4 and result['delay']['summary'] is None
    assert result['delay']['values']==[None,1,2,3,4]
    with pytest.raises(ValueError):s.summarize(rows[:4])


def test_paired_differences_preserve_missing_endpoints_and_original_order():
    assert s.difference(dict(f1=.7,delay=None),dict(f1=.2,delay=4)) == dict(f1=pytest.approx(.5),delay=None)
    with pytest.raises(ValueError):s.difference(dict(f1=.7),dict(f1=.2,extra=1))


def test_seven_dataset_macro_is_within_seed_not_35_independent_repetitions():
    rows=[[{'strict.1.micro.f1':d+seed/10,'strict.1.entity.x.f1':9,'strict.1.event.events':10} for seed in range(5)] for d in range(7)]
    macro=s.macro_vectors(rows)
    assert len(macro)==5 and [v['strict.1.micro.f1'] for v in macro] == pytest.approx([3,3.1,3.2,3.3,3.4])
    assert s.summarize(macro)['strict.1.micro.f1']['summary']['n']==5
    assert all(list(v)==['strict.1.micro.f1'] for v in macro)
    with pytest.raises(ValueError):s.macro_vectors(rows[:6])


def test_complete_metric_schema_separates_strict_and_oracle_PA_diagnostics():
    metrics=example_metrics();v=s.metrics_vector(metrics)
    assert v['strict.1.micro.f1']==metrics['strict']['1']['micro']['f1']
    assert v['test_label_selected_diagnostic.best_grid_adjust_f_score']==metrics['tab_core_diagnostic']['best_grid_adjust_f_score']
    assert len([k for k in v if k.startswith('test_pooled_diagnostic.')])==20
    assert not any('ci95' in k or 'resamples' in k for k in v)


@pytest.mark.parametrize('mutation',['PA','test_calibration','grid'])
def test_changed_evaluation_protocol_is_rejected(mutation):
    metrics=example_metrics()
    if mutation=='PA':metrics['strict']['1']['point_adjustment']=True
    elif mutation=='test_calibration':metrics['strict']['1']['calibration']['calibration_split']='test'
    else:metrics['tab_core_diagnostic']['ratio_grid'].pop()
    with pytest.raises(ValueError):s.metrics_vector(metrics)


def test_real105_original_failures_do_not_become_positive_repair_results():
    source=s.read(s.ROOT/s.PRODUCER);cases=all_cases();selected=s.tasks(cases)
    affected=[t for t in selected.values() if any(i in source['preserved_original_failures'] for i in t['cases'])]
    assert len(source['preserved_original_failures'])==105 and len(affected)==55
    cohort=[t for t in affected if t['kind']=='cohort'];assert len(cohort)==21
    assert all(cases[t['cases'][0]]['family']=='original' for t in cohort)
    assert all(any(i in source['preserved_original_failures'] for i in t['cases']) for t in affected)


def test_missing_proofs_wait_but_actual_failed_guard_is_terminal(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s.producer,'source_guard_records',lambda *_:[])
    source=dict(artifact_root=str(tmp_path/'proofs'),state_root='state',preserved_original_failures={},existing_native_certificate=dict(id='old'))
    cases=dict(a=dict(job=dict(id='a'))); task=dict(cases=['a'])
    assert s.readiness(source,cases,task)==('pending',{})
    proof,guard=s.phase_paths(source,'a','native');s.write(guard,dict(passed=False,exit_code=1,resource_guard_aborted=False))
    status,negative=s.readiness(source,cases,task)
    assert status=='unestimable' and negative['a']['outcome']=='preserved_failed_full_verification'


def test_full_case_requires_unchanged_native_arrays_guard_and_source_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    original=tmp_path/s.PRODUCER;original.parent.mkdir(parents=True);original.write_text('{}')
    score=tmp_path/'scores.npz';score.write_bytes(b'fixture artifact hashing, not benchmark scores')
    case=dict(family='original',queue=dict(python='python.exe'),job=dict(id='a',seed=1103))
    binding=dict(result_sha256='fixture',historical_execution_guard_proven=False)
    monkeypatch.setattr(s.producer,'check_source',lambda *_:({'metrics':{}},{},binding))
    source=dict(artifact_root=str(tmp_path/'proofs'),state_root='state',math_tolerance=dict(rtol=2e-6,atol=2e-6),existing_native_certificate=dict(id='old'))
    native=dict(passed=True,job=case['job'],source_binding=binding,runtime_sha256=s.sha(original),all_original_arrays_identical=True,
        all_native_groups_reinferred=True,fit_calls=0,training_updates_performed=0,math_tolerance=source['math_tolerance'],scores_path=str(score),scores_sha256=s.sha(score))
    path,guard=s.phase_paths(source,'a','native');s.write(path,native)
    independent=dict(passed=True,job=case['job'],source_binding=binding,all_original_arrays_and_labels_reviewed=True,
        metrics_events_and_bootstrap_replayed_exactly=True,training_updates_performed=0,existing_native_certificate_reused=False,
        ideal_float64_path_equivalence_certified=False,full_native_split_coverage=[],native_proof_binding=dict(path=str(path),sha256=s.sha(path)))
    ip,ig=s.phase_paths(source,'a','independent');s.write(ip,independent)
    for phase,proof,gpath in [('native',path,guard),('independent',ip,ig)]:
        command=[str(tmp_path/'python.exe'),'-B','-X','utf8','-u','-m',s.producer.MODULE,'--config',s.PRODUCER,'--stage',phase,'--job-id','a']
        s.write(gpath,dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=s.sha(proof),exact_child_handle=dict(command=command)))
    assert s.full_case(source,case)[2]['source_binding']['historical_execution_guard_proven'] is False
    assert s.full_case(source,case)[2]['ideal_float64_path_equivalence_certified'] is False
    score.write_bytes(b'changed coordinate artifact')
    with pytest.raises(ValueError,match='native inference'):s.full_case(source,case)


def test_worker_rejects_mismatched_full_time_support(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'readiness',lambda *_:('ready',{}))
    monkeypatch.setattr(s,'support',lambda dataset,_:dataset['support'])
    cases={}
    ids=[]
    for side in ['left','right']:
        for seed in s.SEEDS:
            identity=f'{side}{seed}';ids.append(identity)
            cases[identity]=dict(job=dict(id=identity,seed=seed,dataset='psm'),family='original')
    metrics=example_metrics()
    def full(_,case):
        identity=case['job']['id'];support='different time coordinates' if identity.startswith('right') else 'source time coordinates'
        return dict(metrics=metrics),dict(input_sha256=support,support=support),dict(full_native_split_coverage=[])
    monkeypatch.setattr(s,'full_case',full)
    with pytest.raises(ValueError,match='matching original seed'):s.worker({}, {}, cases,dict(kind='paired',cases=ids))


def test_support_binds_actual_complete_labels_and_time_coordinates(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.setattr(s,'ROOT',tmp_path)
    p=tmp_path/'input.npz'
    np.savez(p,e_labels=np.array([0,1,1,0]),e_timestamps=np.array([10,11,12,13]),e_source_rows=np.arange(4))
    dataset=dict(input_npz='input.npz',entities=[dict(entity_id='entity',array_prefix='e')])
    proof=dict(full_native_split_coverage=[dict(entity='entity',role='test',points=4)])
    first=s.support(dataset,proof)
    np.savez(p,e_labels=np.array([0,1,1,0]),e_timestamps=np.array([10,11,12,14]),e_source_rows=np.arange(4))
    second=s.support(dataset,proof)
    assert first['entities'][0]['metadata']['_labels']==second['entities'][0]['metadata']['_labels']
    assert first != second


@pytest.mark.parametrize('kind',['cohort','paired','macro'])
def test_full_statistics_worker_fixture_produces_correct_five_seed_analysis(tmp_path,monkeypatch,kind):
    # Mathematical/serialization fixture; no source training or performance claim.
    import numpy as np
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'readiness',lambda *_:('ready',{}))
    np.savez(tmp_path/'data.npz',e_labels=np.array([0,1,1,0]),e_timestamps=np.arange(4))
    cases={};ids=[]
    count={'cohort':1,'paired':2,'macro':7}[kind]
    for group in range(count):
        for index,seed in enumerate(s.SEEDS):
            identity=f'fixture_{group}_{seed}';ids.append(identity)
            cases[identity]=dict(family='fixture',job=dict(id=identity,seed=seed,dataset='fixture'),group=group,index=index)
    def full(_,case):
        metrics=example_metrics();index=case['index'];group=case['group']
        multiplier=2 if kind=='paired' and group==0 else 1
        value=(index+1)*.1*multiplier+(group*.01 if kind=='macro' else 0)
        metrics['strict']['1']['micro']['f1']=value
        dataset=dict(input_sha256='fixture',input_npz='data.npz',entities=[dict(entity_id='entity',array_prefix='e')])
        evidence=dict(id=case['job']['id'],full_native_split_coverage=[dict(entity='entity',role='test',points=4)],
            historical_execution_guard_proven=False,ideal_float64_path_equivalence_certified=False)
        return dict(metrics=metrics),dataset,evidence
    monkeypatch.setattr(s,'full_case',full)
    (tmp_path/'runtime.json').write_text('{}')
    runtime=dict(artifact_root=str(tmp_path/'outputs'),_config_path='runtime.json',primary_endpoint='strict.1.micro.f1')
    task=dict(id='fixture',kind=kind,cases=ids)
    s.worker(runtime,{},cases,task)
    report=s.read(tmp_path/'outputs/fixture/statistics.json')
    primary=report['statistics']['strict.1.micro.f1']['summary']
    assert primary['n']==5 and primary['mean']==pytest.approx(.33 if kind=='macro' else .3)
    assert report['no_new_fit_inference_or_seed'] and not report['strict_TAB_harness_equivalence']
    assert all(not row['historical_execution_guard_proven'] and not row['ideal_float64_path_equivalence_certified'] for row in report['full_case_evidence'].values())
    with pytest.raises(FileExistsError):s.worker(runtime,{},cases,task)


def test_real_registration_pins_all206_and_rejects_changed_policy():
    config=registration();source,cases,selected=s.verify(config)
    assert len(cases)==440 and len(selected)==206 and config['settings']==source['settings']
    altered=deepcopy(config);altered['primary_endpoint']='test_label_selected_diagnostic.best_grid_adjust_f_score'
    with pytest.raises(ValueError):s.verify(altered)


def test_metadata_verification_imports_neither_numpy_nor_torch():
    code='from scripts.flow_matching.register_objective_full440_statistics_v1 import registration;from scripts.flow_matching import run_objective_full440_statistics_v1 as s;import sys;s.verify(registration());assert not any(k in sys.modules for k in ["numpy","torch","pandas"])'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
