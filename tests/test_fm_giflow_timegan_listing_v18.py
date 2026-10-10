import copy
import pytest

from scripts.flow_matching.publish_giflow_timegan_listing_v18 import corrected_updates,pending_timegan,METRICS


def proof():
    return dict(full_test_arrays_and_metrics_passed=True,track='reviewed_corrected',
        job_id='giflow_native__reviewed_corrected__air36__main__seed393',seed=393,actual_full_training_epochs=68,
        actual_optimizer_updates=3264,full_test_batches=14,masked_targets_with_window_repetitions=268613,
        test_used_in_validation=False,author_equivalence_certified=False,
        selection={'tested_weights_match_validation_checkpoint':True},
        native_metrics={m:10. for m in METRICS},independently_recomputed_metrics={m:10. for m in METRICS})


def rows():
    return [dict(algorithm='GiFlow/main',dataset='air36',protocol='giflow_native/reviewed_corrected',metric=m,
                 n=1,required_repeats=5,mean=10.) for m in METRICS]


def test_completed_original_native_scores_are_retained():
    actual=corrected_updates(rows(),proof(),'full-arrays')
    assert len(actual)==4 and all(r['n']==1 and r['mean']==10 for r in actual)
    assert all(r['strict_TAB_result'] is False for r in actual)


@pytest.mark.parametrize('key,value',[('masked_targets_with_window_repetitions',268612),('actual_optimizer_updates',3263),
    ('full_test_batches',13),('full_test_arrays_and_metrics_passed',False),('author_equivalence_certified',True),
    ('test_used_in_validation',True)])
def test_partial_or_unsupported_result_not_promoted(key,value):
    audit=proof();audit[key]=value
    with pytest.raises(ValueError):corrected_updates(rows(),audit,'full-arrays')


def test_changed_native_score_and_reduced_seed_budget_rejected():
    actual=rows();actual[0]['mean']=9.
    with pytest.raises(ValueError):corrected_updates(actual,proof(),'full-arrays')
    actual=rows();actual[0]['required_repeats']=1
    with pytest.raises(ValueError):corrected_updates(actual,proof(),'full-arrays')


def model():
    return {'parameters':dict(seq_len=24,hidden_dim=24,num_layer=3,iterations=50000,batch_size=128,metric_iteration=10)}


def queue():
    return {'jobs':[dict(dataset=d,module=m,seed=s) for d in ('stock','energy','sine')
                    for m in ('gru','lstm','lstmLN') for s in range(1103,1108)]}


def test_all_native_timegan_slots_are_pending_not_zero_performance():
    actual=pending_timegan(model(),queue(),'new','task algorithm dataset protocol metric mean std se ci95_low ci95_high'.split(),'queue')
    assert len(actual)==18
    assert all(r['mean']=='' and r['n']==0 and r['required_repeats']==5 for r in actual)
    assert all(r['evaluator_repeats']==10 for r in actual)


def test_timegan_source_module_and_seed_scope_cannot_shrink():
    actual=queue();actual['jobs']=actual['jobs'][:-1]
    with pytest.raises(ValueError):pending_timegan(model(),actual,'new',['mean'],'queue')
    actual=queue();actual['jobs']=[r for r in actual['jobs'] if r['module']!='lstmLN']
    with pytest.raises(ValueError):pending_timegan(model(),actual,'new',['mean'],'queue')
