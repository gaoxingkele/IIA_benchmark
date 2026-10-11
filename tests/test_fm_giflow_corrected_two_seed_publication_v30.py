import copy
import math
import pytest

from scripts.flow_matching.publish_giflow_corrected_two_seed_listing_v30 import validate_pair,merge_pair,METRICS
from scripts.flow_matching.verify_giflow_corrected_full_replay_v2 import full_coverage

def evidence():
    proofs,results=[],[]
    for seed,targets,value in [(393,268613,10.),(394,264837,12.)]:
        job=dict(id='giflow_native__reviewed_corrected__air36__main__seed'+str(seed),seed=seed,
            track='reviewed_corrected',arguments=dict(batch_size=128,window=24,channel=1))
        batches=[dict(batch=i,windows=128 if i<13 else 83,
            targets=targets//14+(1 if i<targets%14 else 0)) for i in range(14)]
        metrics={key:value for key in METRICS}
        result=dict(job=job,parameters=545987,epochs_completed=68,optimizer_updates=3264,
            data_binding=dict(windows={'test':1747},nodes=36),full_test_batches=14,
            test_targets_with_window_repetitions=targets,artifacts=batches,metrics=metrics,
            selection=dict(test_used_in_validation=False,tested_weights_match_validation_checkpoint=True))
        proof=dict(job_id=job['id'],full_model_replay_and_independent_saved_arrays_verified=True,
            coverage=full_coverage(result,job),additional_training_seed_slots=0,author_equivalence_certified=False,
            original_metrics=metrics,independently_recomputed_metrics=metrics,
            prediction_tolerance=dict(rtol=.0002,atol=.001),metric_tolerance=dict(rel_tol=1e-5,abs_tol=1e-5),
            full_batch_receipts=batches)
        proofs.append(proof)
        results.append(result)
    return proofs,results

def test_two_original_models_use_between_training_seed_uncertainty_not_repeat_inference():
    proofs,results=evidence()
    statistics=validate_pair(proofs,results)
    for item in statistics.values():
        assert item['n']==2 and item['mean']==11.
        assert item['std']==pytest.approx(math.sqrt(2)) and item['se']==pytest.approx(1.)
        assert item['ci95_low']==pytest.approx(11.-12.7062047364)
        assert item['ci95_high']==pytest.approx(11.+12.7062047364)

@pytest.mark.parametrize('field,value',[
    ('full_model_replay_and_independent_saved_arrays_verified',False),('additional_training_seed_slots',1),
    ('author_equivalence_certified',True),('prediction_tolerance',dict(rtol=.01,atol=.01)),
    ('metric_tolerance',dict(rel_tol=.01,abs_tol=.01)),('full_batch_receipts',[])])
def test_partial_models_extra_seeds_relaxed_tolerances_or_missing_full_arrays_rejected(field,value):
    proofs,results=evidence()
    proofs[1][field]=value
    with pytest.raises(ValueError):validate_pair(proofs,results)

def test_other_seed_masks_budgets_selection_and_nonfinite_metrics_rejected():
    proofs,results=evidence()
    with pytest.raises(ValueError):validate_pair(proofs[:1],results[:1])
    for field,value in [('epochs_completed',67),('optimizer_updates',3200),('parameters',100),
        ('test_targets_with_window_repetitions',268613),('selection',dict(test_used_in_validation=True,tested_weights_match_validation_checkpoint=True)),
        ('metrics',{k:float('nan') for k in METRICS})]:
        changed=copy.deepcopy(results)
        changed[1][field]=value
        with pytest.raises(ValueError):validate_pair(proofs,changed)

def test_only_four_existing_corrected_metric_rows_change_and_all_other_metadata_remain_exact():
    proofs,results=evidence()
    rows=[dict(algorithm='GiFlow/main',dataset='air36',protocol='giflow_native/reviewed_corrected',
        task='imputation',metric=k,n='1',required_repeats='5',pattern='point',missing_ratio='0.2',
        mean='10.0',captured_utc='old',source='old',epochs='',score_recipe='') for k in METRICS]
    other=dict(rows[0],dataset='aqi',n='0',mean='',captured_utc='keep_exact')
    original=rows+[other]
    merged,changed=merge_pair(original,validate_pair(proofs,results),'new','evidence')
    assert changed==[0,1,2,3] and merged[4]==other and original[0]['n']=='1'
    assert all(r['n']=='2' and r['required_repeats']=='5' for r in merged[:4])
    with pytest.raises(ValueError):merge_pair(original[:-2],validate_pair(proofs,results),'new','evidence')
    bad=copy.deepcopy(original)
    bad[0]['required_repeats']='2'
    with pytest.raises(ValueError):merge_pair(bad,validate_pair(proofs,results),'new','evidence')
