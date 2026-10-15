"""Complete cohorts,seed pairing,common support and descriptive uncertainty."""
from copy import deepcopy
import math
import os
import statistics
import subprocess
import sys

import numpy as np
import pytest

from scripts.flow_matching import run_anomaly_transformer_full_cohort_statistics_v1 as subject
from scripts.flow_matching.register_anomaly_transformer_full_cohort_statistics_v1 import registration


def cases_fixture():
    result = {}
    for family, (path, _, _) in subject.producer.FAMILIES.items():
        config = subject.read(subject.ROOT/path)
        for job in config['jobs']:
            result[job['id']] = dict(family=family, path=path, config=config, job=job)
    return result


def test_all215_statistics_tasks_cover500_models_and_every_required_dataset_seed():
    cases = cases_fixture(); selected = subject.tasks(cases)
    assert len(selected) == 215
    assert {k:sum(t['kind'] == k for t in selected.values()) for k in ['cohort','paired','macro']} == dict(cohort=100,paired=95,macro=20)
    assert {i for t in selected.values() if t['kind']=='cohort' for i in t['cases']} == set(cases)
    for t in selected.values():
        assert len(t['cases']) == {'cohort':5,'paired':10,'macro':25}[t['kind']]
        if t['kind'] == 'paired':
            assert [cases[i]['job']['seed'] for i in t['cases'][:5]] == [cases[i]['job']['seed'] for i in t['cases'][5:]] == subject.SEEDS


def test_no_duplicate_or_missing_seed_is_accepted_as_a_complete_cohort():
    cases = cases_fixture(); cases.pop(next(iter(cases)))
    with pytest.raises(ValueError): subject.tasks(cases)
    cases = cases_fixture(); first = next(iter(cases)); cases['duplicate'] = cases[first]
    with pytest.raises(ValueError): subject.tasks(cases)


def test_df4_t_critical_and_five_seed_standard_error_are_independently_correct():
    from scipy.stats import t
    values = [.1,.2,.3,.4,.5]
    r = subject.five_seed_summary(values)
    expected_mean = sum(values)/5
    variance = sum((v-expected_mean)**2 for v in values)/4
    assert abs(subject.DF4_CRITICAL_975-float(t.ppf(.975,4))) < 1e-12
    assert r['mean'] == pytest.approx(.3) and r['sample_standard_deviation'] == pytest.approx(math.sqrt(variance))
    assert r['standard_error'] == pytest.approx(math.sqrt(variance/5))
    assert r['student_t_df4_95_interval'] == pytest.approx([.3-float(t.ppf(.975,4))*math.sqrt(variance/5),.3+float(t.ppf(.975,4))*math.sqrt(variance/5)])
    assert r['leave_one_seed_out_means']['1103'] == pytest.approx(.35)
    assert r['leave_one_seed_out_means']['1107'] == pytest.approx(.25)


@pytest.mark.parametrize('values,seeds', [([1,2,3,4],subject.SEEDS),([1,2,3,4,5],[1103]*5),
    ([1,2,3,4,float('nan')],subject.SEEDS),([1,2,3,4,True],subject.SEEDS)])
def test_rejects_partial_duplicate_or_invalid_seed_observations(values,seeds):
    with pytest.raises(ValueError): subject.five_seed_summary(values,seeds)


def test_intervals_are_not_clipped_and_robustness_is_not_a_new_training_seed():
    r = subject.five_seed_summary([0,0,0,0,1])
    assert r['student_t_df4_95_interval'][0] < 0
    assert r['n'] == 5 and len(r['leave_one_seed_out_means']) == 5
    assert r['median'] == 0 and r['minimum'] == 0 and r['maximum'] == 1
    assert not r['hypothesis_test_or_generalization_guarantee']


def test_all_numeric_controls_remain_separate_and_missing_values_are_not_dropped():
    trees = [dict(no_PA=dict(f1=i/10),paper_PA=dict(f1=.9),roc=[dict(r_percent=.5,fpr=i/20)],undefined=None,flag=False) for i in range(5)]
    rows = subject.aggregate_metric_trees(trees)
    by_path = {tuple(r['metric_path']):r for r in rows}
    assert by_path[('no_PA','f1')]['summary']['mean'] == pytest.approx(.2)
    assert by_path[('paper_PA','f1')]['summary']['mean'] == .9
    assert by_path[('roc',0,'r_percent')]['summary']['values'] == [.5]*5
    assert by_path[('undefined',)]['summary'] is None and ('flag',) not in by_path
    with pytest.raises(ValueError): subject.aggregate_metric_trees(trees[:4])
    bad = deepcopy(trees); bad[0]['undefined'] = 0
    with pytest.raises(ValueError): subject.aggregate_metric_trees(bad)


def test_paired_common_prefix_preserves_full_validation_and_avoids_different_truth_denominators(tmp_path):
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import independent_evaluation
    contract = dict(paper_evaluation=dict(r_percent=dict(SMD=1),fixed_delta=dict(SMD=.1),ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]))
    labels = np.array([0,1,1,0,1,1],np.int8); validation = np.array([.5,.5],np.float32)
    cases, baselines, results, baseline_results = [], [], [], []
    for family, target_cases, target_results, scores in [
        ('sensitivity',cases,results,np.array([0,1,0,0,5,5],np.float32)),
        ('main',baselines,baseline_results,np.array([0,.2,.3,0],np.float32))]:
        cache = tmp_path/family/'cache/SMD'; cache.mkdir(parents=True)
        np.save(cache/'labels.npy',labels)
        for seed in subject.SEEDS:
            job = dict(id=f'{family}_{seed}',dataset='SMD',seed=seed)
            config = dict(artifact_root=str(tmp_path/family/'models'),data_cache_root=str(tmp_path/family/'cache'),state_root='fixture')
            case = dict(family=family,job=job,config=config); target_cases.append(case)
            base,_,_ = subject.producer.source_paths(case); base.mkdir(parents=True)
            np.savez(base/'test_scores.npz',**{'all.multiplication':scores})
            np.savez(base/'validation_scores.npz',**{'all.multiplication':validation})
            metric = independent_evaluation(labels[:len(scores)],scores,validation,1,.1,[.5,1,1.5,2,10,20,30])
            target_results.append(dict(covered_points=dict(test=len(scores)),metrics={'all.multiplication':metric}))
    differences, evidence = subject.paired_common_support(results,baseline_results,cases,baselines,contract)
    assert differences['validation_ratio.no_PA.f1']['mean'] == pytest.approx(2/3)
    assert all(r['common_test_points']==4 and r['variant_original_test_points']==6 and r['baseline_original_test_points']==4 for r in evidence)
    assert all(r['variant_metrics']['validation_ratio']['threshold']==.5 for r in evidence)
    assert results[0]['metrics']['all.multiplication']['validation_ratio']['no_PA']['f1'] > 2/3
    changed = deepcopy(baselines); changed[0]['job']['seed'] = 1104
    with pytest.raises(ValueError): subject.paired_common_support(results,baseline_results,cases,changed,contract)


def test_macro_uses_five_seed_dataset_means_instead_of_25_independent_samples():
    reports, cases = [], {}
    for d, dataset in enumerate(subject.DATASETS):
        rows = []
        for i,seed in enumerate(subject.SEEDS):
            value=d/10+i/100; identity=f'{dataset}_{seed}';cases[identity]=dict(job=dict(seed=seed))
            metric = {protocol:{adjustment:{k:value for k in ['precision','recall','f1']} for adjustment in ['no_PA','paper_PA']} for protocol in ['validation_ratio','fixed_delta']}
            metric.update(continuous_unadjusted_score_AUROC=value,continuous_score_AP=value)
            rows.append(dict(id=identity,seed=seed,metrics={'all.multiplication':metric}))
        reports.append(dict(dataset=dataset,variant='main',per_seed_metrics=rows))
    by_seed, summaries = subject.macro_summary(reports,cases)
    assert len(by_seed)==5 and by_seed[0]['metrics']['all.multiplication']['validation_ratio.no_PA.f1'] == pytest.approx(.2)
    assert all(r['summary']['n']==5 for r in summaries)
    assert summaries[0]['summary']['mean'] == pytest.approx(.22)
    with pytest.raises(ValueError): subject.macro_summary(reports[:4],cases)
    bad = deepcopy(reports);bad[0]['per_seed_metrics'][0]['seed']=1104
    with pytest.raises(ValueError): subject.macro_summary(bad,cases)


def test_readiness_needs_all_five_real_proofs_and_passed_guards(tmp_path,monkeypatch):
    monkeypatch.setattr(subject,'ROOT',tmp_path)
    source=dict(artifact_root=str(tmp_path/'producer'),state_root='producer_state')
    task=dict(kind='cohort',id='fixture',cases=[f'model_{i}' for i in range(5)])
    for identity in task['cases'][:4]:
        for proof,receipt in subject.producer_proof_paths(source,identity):
            subject.write(proof,dict(passed=True));subject.write(receipt,dict(passed=True,exit_code=0,resource_guard_aborted=False))
    assert subject.readiness({},source,task,set(),{}) is False
    for proof,receipt in subject.producer_proof_paths(source,task['cases'][4]):
        subject.write(proof,dict(passed=True));subject.write(receipt,dict(passed=True,exit_code=0,resource_guard_aborted=False))
    assert subject.readiness({},source,task,set(),{}) is True
    subject.write(tmp_path/'producer_state/status.json',dict(failed_preserved={'model_4':'source guard failed'}))
    with pytest.raises(ValueError,match='failed original producer'):subject.readiness({},source,task,set(),{})


def test_statistics_worker_binds_all_five_guarded_proofs_and_rejects_changed_native_artifacts(tmp_path,monkeypatch):
    # Metadata fixture only. The real producer check is not replaced in production.
    monkeypatch.setattr(subject,'ROOT',tmp_path)
    subject.write(tmp_path/subject.PRODUCER,dict(fixture_only=True))
    relative='stats_fixture.json';subject.write(tmp_path/relative,dict(fixture_only=True))
    source=dict(python=sys.executable,artifact_root=str(tmp_path/'native'),state_root='native_state')
    runtime=dict(artifact_root=str(tmp_path/'summaries'),state_root='stats_state',_config_path=relative)
    cases={};records={}
    for i,seed in enumerate(subject.SEEDS):
        identity=f'fixture_{seed}';job=dict(id=identity,seed=seed,dataset='SMD')
        case=dict(family='main',path='fixture_config.json',config={},job=job);cases[identity]=case
        metrics={'all.multiplication':dict(validation_ratio=dict(no_PA=dict(f1=i/10),paper_PA=dict(f1=.9)))}
        binding=dict(configuration_sha256='fixture_config',result_sha256='fixture_result')
        result=dict(metrics=metrics,data_manifest_sha256='fixture_data',covered_points=dict(test=100),omitted_tail_points=dict(test=5))
        records[identity]=(result,{},binding)
        artifact=tmp_path/'fixture_arrays'/f'{identity}.bin';artifact.parent.mkdir(exist_ok=True);artifact.write_bytes(b'fixture all-array bytes')
        paths=subject.producer_proof_paths(source,identity)
        native=dict(passed=True,job=job,source_binding=binding,runtime_sha256=subject.sha(tmp_path/subject.PRODUCER),
            all_covered_points_reinferred=True,native_dtype_shape_content_identical=True,training_updates_performed=0,
            records=[dict(replay_path=str(artifact),replay_sha256=subject.sha(artifact))])
        subject.write(paths[0][0],native)
        review=dict(passed=True,job=job,source_binding=binding,native_replay_sha256=subject.sha(paths[0][0]),
            all_original_streams_and_covered_arrays_checked=True,
            all_original_thresholds_confusion_counts_PA_and_ranking_metrics_checked=True,
            test_used_for_calibration=False,strict_TAB_result=False,independent_metrics=metrics)
        subject.write(paths[1][0],review)
        for phase,(proof,receipt) in zip(['native','independent'],paths):
            command=[sys.executable,'-B','-X','utf8','-u','-m',subject.producer.MODULE,'--config',subject.PRODUCER,'--stage',phase,'--job-id',identity]
            subject.write(receipt,dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=subject.sha(proof),
                exact_child_handle=dict(command=command),source_configuration='fixture_config.json',source_configuration_sha256='fixture_config'))
    monkeypatch.setattr(subject.producer,'check_source',lambda c,k:records[c['job']['id']])
    task=dict(kind='cohort',id='fixture_cohort',variant='main',dataset='SMD',cases=list(cases))
    subject.worker(runtime,source,cases,{},task,{task['id']:task})
    output,_=subject.stage_paths(runtime,task['id']);report=subject.read(output)
    assert report['passed'] and len(report['model_evidence'])==5
    assert report['summaries'][0]['summary']['n']==5 and report['summaries'][0]['summary']['mean']==pytest.approx(.2)
    assert report['training_updates_performed']==0 and not report['strict_TAB_result']
    first=next(iter(cases));(tmp_path/'fixture_arrays'/f'{first}.bin').write_bytes(b'changed after review')
    with pytest.raises(ValueError,match='artifact changed'):subject.read_full_model(source,cases[first],{})


@pytest.mark.parametrize('change',['partial','support','calibration','macro','interval','guard','endpoint'])
def test_registration_rejects_shrunk_or_changed_statistics_policy(change):
    runtime=registration()
    if change=='partial': runtime['required_models']=495
    elif change=='support': runtime['paired_test_support']='different_original_support'
    elif change=='calibration': runtime['paired_calibration']='test_percentile'
    elif change=='macro': runtime['macro_weighting']='pool_all_dataset_points'
    elif change=='interval': runtime['student_t_critical_df4_975']=1.96
    elif change=='guard': runtime['settings']['minimum_commit_headroom_bytes']=32*1024**3
    else:runtime['primary_endpoint']='best_test_F1'
    with pytest.raises(ValueError):subject.verify(runtime)


def test_full_valid_registration_stays_numerically_cold_in_a_fresh_interpreter():
    code=('import sys;from scripts.flow_matching.register_anomaly_transformer_full_cohort_statistics_v1 import registration;'
        'from scripts.flow_matching.run_anomaly_transformer_full_cohort_statistics_v1 import verify;'
        's,c,k,t=verify(registration());assert len(c)==500 and len(t)==215;'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();assert not any(n in sys.modules for n in ["numpy","pandas","torch"])')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=subject.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr[-4000:]
