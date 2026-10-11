import copy

import pytest

from scripts.flow_matching.publish_spectral_stocks_seed42_listing_v28 import (
    CANONICAL,METRICS,updates,check_increment,validated_native)


def full_native():
    metrics={k:dict(raw_repeats=[.1]*5,evaluator_repeats=5,independent_training_models=1,
        planned_original_training_models=5,between_training_seed_STD=None,between_training_seed_CI95=None)
        for k in METRICS}
    proof=dict(passed=True,canonical_id=CANONICAL,seed=42,original_full_registered_generation_jobs=70,
        actual_full_native_model_count=1,planned_original_training_seed_count=5,actual_model_parameters=294390,
        paper_reported_parameters=291318,original_optimizer_updates=10000,original_gradient_accumulation=2,
        original_training_minibatches=20000,complete_reference_shape=[3662,24,6],complete_generated_shape=[4002,24,6],
        evaluated_generated_rows=3662,actual_full_original_checkpoint_tensors_restored_bitwise=True,
        complete_original_reference_array_bitwise_verified=True,all_original_generated_arrays_finite=True,
        all_four_five_repeat_evaluator_statistics_independently_recomputed=True,
        learned_evaluators_independently_retrained=False,generative_model_reinferred_by_this_audit=False,
        additional_training_seed_slots=0,author_equivalence_certified=False,metrics=metrics)
    result=dict(job={'id':CANONICAL},status='completed',benchmark_smoke=False,
        metrics={k:dict(raw_repeats=[.1]*5) for k in METRICS})
    receipt=dict(id=CANONICAL,exitcode=0,full_result_validated=True,resource_guard_aborted=False)
    return proof,result,receipt


@pytest.mark.parametrize('fault',['budget','batch','reference','capacity','seed','five_models','fake_variance','partial_evaluators','smoke','aborted'])
def test_only_actual_complete_original_generator_is_publishable(fault):
    proof,result,receipt=full_native()
    validated_native(proof,result,receipt)
    if fault=='budget': proof['original_optimizer_updates']=9900
    elif fault=='batch': proof['complete_generated_shape']=[2001,24,6]
    elif fault=='reference': proof['complete_reference_shape']=[3661,24,6]
    elif fault=='capacity': proof['actual_model_parameters']=291318
    elif fault=='seed': proof['seed']=43
    elif fault=='five_models': proof['actual_full_native_model_count']=5
    elif fault=='fake_variance': proof['metrics']['context_fid']['between_training_seed_STD']=.01
    elif fault=='partial_evaluators': proof['metrics']['context_fid']['evaluator_repeats']=4
    elif fault=='smoke': result['benchmark_smoke']=True
    else: receipt['resource_guard_aborted']=True
    with pytest.raises(ValueError): validated_native(proof,result,receipt)


def test_only_four_original_empty_slots_change_and_seed_uncertainty_stays_missing():
    original=[dict(task='unconditional_time_series_generation',algorithm='Spectral-author/diffusion_ts',
        dataset='stocks',protocol='released_author_data',metric=k,n='0',required_repeats='5',mean='',
        paper_mean='.2',captured_utc='old') for k in METRICS]
    original.append(dict(task='anomaly_detection',algorithm='FM',dataset='MSL',protocol='validation',
        metric='f1',n='5',required_repeats='5',mean='.7',captured_utc='old-other'))
    proof={'captured_utc':'new'}
    result={'metrics':{k:dict(mean=.1,std=.01,se=.004,ci95_low=.08,ci95_high=.12) for k in METRICS}}
    changed=updates(original,proof,result,'full-native-proof')
    assert all(r['n']==1 and r['evaluator_repeats']==5 and r['required_repeats']=='5'
        and all(r[key]=='' for key in ['std','se','ci95_low','ci95_high']) for r in changed)
    rows=changed+[copy.deepcopy(original[-1])]
    assert check_increment(original,rows,changed)==[0,1,2,3]
    rows[-1]['mean']='.8'
    with pytest.raises(ValueError): check_increment(original,rows,changed)
    with pytest.raises(ValueError): check_increment(original,rows[:-1],changed)
    bad=copy.deepcopy(original)
    bad[0]['required_repeats']='1'
    with pytest.raises(ValueError): updates(bad,proof,result,'proof')
