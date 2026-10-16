import copy
import math
import struct
import zipfile
import pytest
from scripts.flow_matching import collect_trajectorynet_two_protocols_full270_v1 as s


def npy(path,values,shape=(2,1),dtype='<f4'):
    header=(repr(dict(descr=dtype,fortran_order=False,shape=shape))+'\n').encode('latin1')
    width='f' if dtype=='<f4' else 'd'
    path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+width*len(values),*values))


def test_complete_array_file_identity_detects_signed_zero_shape_and_payload_corruption(tmp_path):
    a,b=tmp_path/'a.npy',tmp_path/'b.npy';npy(a,[0.,1.]);npy(b,[0.,1.])
    assert s.paired_array_identity(a,b,[2,1],'<f4')['complete_file_bytes_identical']
    npy(b,[-0.,1.])
    with pytest.raises(ValueError,match='byte-for-byte'):s.paired_array_identity(a,b,[2,1],'<f4')
    with pytest.raises(ValueError,match='shape'):s.npy_identity(a,[1,2],'<f4')
    b.write_bytes(a.read_bytes()[:-1])
    with pytest.raises(ValueError,match='Truncated'):s.npy_identity(b,[2,1],'<f4')


def torch_zip(path,prefix,data=b'weights'):
    with zipfile.ZipFile(path,'w') as out:
        out.writestr(prefix+'/data.pkl',b'pickle deliberately never executed')
        out.writestr(prefix+'/data/0',data)
        out.writestr(prefix+'/version',b'3\n')


def test_original_selected_step_payload_comparison_ignores_only_zip_root_name(tmp_path):
    a,b=tmp_path/'best.pt',tmp_path/'step.pt'
    torch_zip(a,'checkpt');torch_zip(b,'checkpt-0100')
    assert s.verify_selected_step(a,b)['normalized_zip_payloads_equal']
    torch_zip(b,'checkpt-0100',b'other weights')
    with pytest.raises(ValueError,match='claimed original'):s.verify_selected_step(a,b)


def record():
    result=dict(metrics=dict(W1_future=1.5,W1_past=1.5,W2_future=2.,W2_past=2.),
        full_population=True,primary_direction='past_to_leftout',selected_better_direction=False,primary_W1=1.5,proofs=[])
    for direction,n in [('future',4),('past',2)]:
        result['proofs'].append(dict(direction=direction,shape=[n,3],all_cost_entries_checked=n*3,
            native_W2=dict(shape=[n,3],result_code=1,reported_W2=2.,transport_objective=4.),
            raw_W1=1.5,independent_float64_W1=1.5,native_W1_LP_result_code=1,independent_W1_LP_result_code=1,
            maximum_cost_entry_difference=1e-6,objective_difference=0.,cost_perturbation_bound=1e-6+1.5e-8))
    return result


def test_full_native_W2_raw_W1_and_independent_LP_bounds_cannot_be_swapped_or_sampled():
    counts={'0':2,'1':3,'2':4};r=record();s.metric_proof(r,1,counts)
    wrong=copy.deepcopy(r);wrong['proofs'][0]['all_cost_entries_checked']=3
    with pytest.raises(ValueError,match='every cost-entry'):s.metric_proof(wrong,1,counts)
    wrong=copy.deepcopy(r);wrong['proofs'][1]['cost_perturbation_bound']=100.
    with pytest.raises(ValueError,match='not be interchanged'):s.metric_proof(wrong,1,counts)
    wrong=copy.deepcopy(r);wrong['proofs'][0]['native_W2']['reported_W2']=4.
    with pytest.raises(ValueError,match='not be interchanged'):s.metric_proof(wrong,1,counts)
    wrong=copy.deepcopy(r);wrong['proofs'][0]['independent_W1_LP_result_code']=3
    with pytest.raises(ValueError,match='exact-LP'):s.metric_proof(wrong,1,counts)


def paired():
    rows=[]
    for profile in s.QUEUES:
        for seed in s.SEEDS:
            for leftout in s.LEFT:
                value=seed-42+leftout*10+(1 if profile=='paper_text' else 0)
                rows.append(dict(profile=profile,seed=seed,leftout=leftout,
                    metrics={selection:{metric:value+(.5 if selection==s.SELECTIONS[1] else 0) for metric in s.METRICS}
                        for selection in s.SELECTIONS}))
    return rows


def test_statistics_use_paired_five_seed_means_not_fifteen_unmatched_replicates():
    result=s.summaries(paired());r=result['cohorts']['author_log'][s.SELECTIONS[0]]['W1_past']
    assert r['mean']==22 and math.isclose(r['SD'],math.sqrt(2.5))
    assert r['separate_all15_model_SD']>r['SD']
    contrasts=result['paired_contrasts']
    assert len(contrasts)==16
    a=contrasts['paper_text_minus_author_log__'+s.SELECTIONS[0]+'__W1_past']
    assert a['mean']==1 and a['SD']==0 and a['descriptive_df4_t95']==[1,1]
    b=contrasts['author_log__corrected_minimum_minus_source_final__W1_past']
    assert b['mean']==-.5
    assert not result['test_parameter_selection'] and not result['formal_hypothesis_testing_or_causal_claim']


def test_missing_seed_leftout_duplicate_pair_and_partial_cohort_never_get_statistics():
    rows=paired()
    with pytest.raises(ValueError):s.summaries(rows[:-1])
    rows[-1]=rows[0]
    with pytest.raises(ValueError):s.summaries(rows)
    with pytest.raises(ValueError):s.uncertainty([1.,2.,3.,4.])


def test_full270_scope_and_all_checkpoint_selections_remain_preregistered():
    c=s.registration()
    assert len(c['cases'])==c['required_comparisons']==9
    assert c['required_models']==270 and c['required_checkpoint_selections']==540
    assert c['pending_energy_models_in_two_protocols']==90
    assert sum(len(case['jobs']['author_log'])+len(case['jobs']['paper_text']) for case in c['cases'])==270
    assert c['training_seeds']==[42,43,44,45,46] and c['all_leftouts']==[1,2,3]
    assert c['checkpoint_selections']==s.SELECTIONS and c['primary_metric']=='W1_past'
    assert not c['partial_cohorts_allowed'] and not c['test_coefficient_selection_allowed']
    assert not c['individual_activation_decay_effect_identifiable']
    assert not c['numerical_inference_performed']
