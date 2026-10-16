from copy import deepcopy
import pytest
from scripts.flow_matching import summarize_otcfm_Multiome_full120_paper_rows_v1 as s


def test_every_original120case_belongs_to_one_full_ten_model_row():
    c=s.registration();ids=[i for g in c['groups'] for i in g['required_case_ids']]
    assert len(c['groups'])==12 and len(ids)==len(set(ids))==120
    assert all(len(g['required_case_ids'])==10 and g['leaveout_indices']==[1,2] for g in c['groups'])
    assert c['no_table_result_from_partial_rows'] and c['all_source_and_full_native_guards_required']
    assert c['settings']==s.read(s.ROOT/'configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json')['settings']


def test_real_native_full36_report_checks_legacy_actual_and_selected_updates_separately():
    q=s.read(s.ROOT/s.native.QUEUE);rc=s.read(s.ROOT/s.replay.CONFIG);j=q['jobs'][0]
    source,provenance=s.inspect(q,j);_,gp,rp=s.generic.guard_paths(q,rc,j);r,g=s.read(rp),s.read(gp)
    assert source['optimizer_updates']!=source['epoch_history'][-1]['optimizer_updates']
    s.validate_native(j,source,provenance,r,g,s.sha(rp),s.sha(s.ROOT/s.replay.CONFIG))
    for fault in ['wrong_total','wrong_selected','partial_metrics','missing_timepoint','nonoptimal','wrong_population','objective_bound','failed_guard']:
        a,b,c=deepcopy(source),deepcopy(r),deepcopy(g)
        if fault=='wrong_total':a['epoch_history'][-1]['optimizer_updates']+=1
        elif fault=='wrong_selected':a['optimizer_updates']+=1
        elif fault=='partial_metrics':b['native_metrics'].pop(next(iter(b['native_metrics'])))
        elif fault=='missing_timepoint':b['full_test_population_distances_independently_recomputed'][0]['solves'].pop()
        elif fault=='nonoptimal':b['full_test_population_distances_independently_recomputed'][0]['solves'][0]['native_LP_result_code']=0
        elif fault=='wrong_population':b['full_test_population_distances_independently_recomputed'][0]['solves'][0]['full_cost_shape'][0]-=1
        elif fault=='objective_bound':b['full_test_population_distances_independently_recomputed'][0]['solves'][0]['objective_absolute_difference']=1
        else:c['passed']=False
        with pytest.raises(ValueError):s.validate_native(j,a,provenance,b,c,s.sha(rp),s.sha(s.ROOT/s.replay.CONFIG))


def test_partial_five_seed_single_leaveout_is_not_a_complete_paper_row():
    c=s.registration();group=c['groups'][0];q=s.read(s.ROOT/s.native.QUEUE)
    records=[dict(id=j['id'],dataset=j['dataset'],method=j['method'],protocol=j['protocol'],seed=j['seed'],
        leaveout_timepoint=j['leaveout_timepoint'],heldout_W1=.5) for j in q['jobs'] if j['id'] in group['required_case_ids']]
    complete=s.generic.summarize(records,group)
    assert complete['independently_trained_models']==10 and complete['training_seeds']==5
    assert not complete['historical_paper_equivalence_certified']
    with pytest.raises(ValueError):s.generic.summarize([r for r in records if r['leaveout_timepoint']==1],group)
