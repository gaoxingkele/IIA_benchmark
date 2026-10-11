import copy
import math
import statistics
import pytest
from scripts.flow_matching.summarize_wt_full_image_cohort_v1 import (
    checked_records, build_summary, review_summary, independent_statistics)
from scripts.flow_matching.wt_full_cohort_summary_contract_v1 import METRICS, SEEDS, STEPS, UNCERTAINTY


def fixture():
    names = ['category_%02d' % i for i in range(15)]
    counts = [100+i for i in range(14)]; counts.append(1725-sum(counts))
    paper = {name: 80.+i for i, name in enumerate(METRICS)}
    records = []
    for seed in SEEDS:
        for step in STEPS:
            categories = [dict(category=name, original_test_images=counts[i], metrics={
                metric:.25+i*.01+(seed-1103)*.005+step*.0001+j*.001 for j, metric in enumerate(METRICS)})
                for i, name in enumerate(names)]
            records.append(dict(seed=seed, steps=step, checkpoint_sha256='checkpoint_%d'%seed,
                training_binding=dict(job_sha256='job_%d'%seed, dataset_configuration_sha256='data',
                    model_configuration_sha256='model', optimizer_configuration_sha256='optimizer'),
                paper_reported=paper, categories=categories,
                macro={metric:statistics.mean(row['metrics'][metric] for row in categories) for metric in METRICS}))
    return records, names, counts, paper


def test_complete_all_six_step_five_seed_category_and_pair_statistics_verified_independently():
    records, names, counts, paper = fixture()
    result = build_summary(records, names, counts, paper)
    assert review_summary(result, records, names, counts, paper)
    assert len(result['aggregate_metrics']) == 42 and len(result['per_category_metrics']) == 630
    assert len(result['paired_step_differences']) == 84
    assert all(row['n'] == 5 for row in result['aggregate_metrics'])
    assert UNCERTAINTY['statistical_unit'] == 'independent_original_training_seed'
    zero = [row for row in result['paired_step_differences'] if row['steps'] == row['reference_steps']]
    assert len(zero) == 14 and all(row['mean'] == row['std'] == row['se'] == 0. for row in zero)
    example = next(row for row in result['paired_step_differences'] if row['steps']==50 and row['reference_steps']==1)
    assert math.isclose(example['mean'], .0049, abs_tol=1e-12)


@pytest.mark.parametrize('action', ['remove', 'duplicate', 'different_checkpoint', 'different_data',
    'repeated_training_job', 'fake_macro', 'category_order', 'category_count', 'short_category', 'NaN', 'paper_value'])
def test_partial_unpaired_or_incorrect_native_category_results_are_rejected(action):
    records, names, counts, paper = fixture()
    if action=='remove': records.pop()
    elif action=='duplicate': records[-1] = copy.deepcopy(records[0])
    elif action=='different_checkpoint': records[1]['checkpoint_sha256']='different'
    elif action=='different_data':
        for row in records:
            if row['seed']==1104: row['training_binding']['dataset_configuration_sha256']='changed'
    elif action=='repeated_training_job':
        for row in records:
            if row['seed']==1104: row['training_binding']['job_sha256']='job_1103'
    elif action=='fake_macro': records[0]['macro']['image_AUROC']+=.01
    elif action=='category_order': records[0]['categories'].reverse()
    elif action=='category_count': records[0]['categories'][0]['original_test_images']-=1
    elif action=='short_category': records[0]['categories'].pop()
    elif action=='NaN': records[0]['categories'][0]['metrics']['image_AUROC']=float('nan')
    elif action=='paper_value': records[0]['paper_reported']=dict(paper,image_AUROC=99.)
    with pytest.raises(ValueError): checked_records(records,names,counts,paper)


@pytest.mark.parametrize('field,key,value', [('aggregate_metrics','n',30), ('aggregate_metrics','std',0),
    ('aggregate_metrics','ci95_low',0), ('aggregate_metrics','paper_percent',99),
    ('aggregate_metrics','author_equivalence_certified',True), ('aggregate_metrics','difference_percentage_points',0),
    ('per_category_metrics','original_test_images',1), ('paired_step_differences','mean',1),
    ('paired_step_differences','uncertainty_is_significance_test',True),
    ('fixed_grid_descriptive_robustness','step_selected_for_final_performance',True)])
def test_independent_review_rejects_false_uncertainty_pairing_paper_or_best_step_claim(field,key,value):
    records,names,counts,paper=fixture(); candidate=build_summary(records,names,counts,paper)
    candidate[field][0][key]=value
    with pytest.raises(ValueError): review_summary(candidate,records,names,counts,paper)


def test_independent_t_interval_matches_known_five_sample_uncertainty_without_clipping():
    result=independent_statistics([0.,1.,2.,3.,4.])
    assert result['n']==5 and result['mean']==2.
    assert math.isclose(result['std'],math.sqrt(2.5),rel_tol=1e-14)
    assert math.isclose(result['se'],math.sqrt(.5),rel_tol=1e-14)
    assert math.isclose(result['ci95_low'],2.-2.7764451051977987*math.sqrt(.5),rel_tol=1e-12)
    with pytest.raises(ValueError): independent_statistics([.5]*30)


def test_full_result_without_all_category_or_metric_summaries_cannot_pass_review():
    records,names,counts,paper=fixture(); result=build_summary(records,names,counts,paper)
    for field in ['aggregate_metrics','per_category_metrics','paired_step_differences','fixed_grid_descriptive_robustness']:
        wrong=copy.deepcopy(result);wrong[field].pop()
        with pytest.raises(ValueError): review_summary(wrong,records,names,counts,paper)


def test_proof_locations_are_eligibility_only_and_summary_is_missing_until_review(tmp_path, monkeypatch):
    import scripts.flow_matching.wt_full_cohort_summary_contract_v1 as contract
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    queue=dict(cases=[dict(output_directory='case_%d'%i) for i in range(30)])
    config=dict(output_directory='summary')
    assert not contract.score_locations_ready(config,queue)
    assert not contract.complete_summary(config)
    for case in queue['cases']:
        path=tmp_path/case['output_directory'];path.mkdir();(path/'fully_verified_score_result.json').write_text('{}')
    assert contract.score_locations_ready(config,queue)
    # Thirty file names alone are not independent completed-case or summary evidence.
    assert not contract.complete_summary(config)
