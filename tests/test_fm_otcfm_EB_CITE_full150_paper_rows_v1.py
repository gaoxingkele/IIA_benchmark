import ast
from copy import deepcopy
import inspect
import itertools
import math
import pytest
from scripts.flow_matching import summarize_otcfm_EB_CITE_full150_paper_rows_v1 as s


def fixture(dataset):
    leaveouts=[1,2,3] if dataset=='released_author_EB' else [1,2]
    records=[dict(id=f'fixture_{seed}_{left}',dataset=dataset,method='cfm',protocol='released_yaml',
        seed=seed,leaveout_timepoint=left,heldout_W1=seed/100+left/10)
        for seed,left in itertools.product(range(42,47),leaveouts)]
    group=dict(dataset=dataset,method='cfm',protocol='released_yaml',leaveout_indices=leaveouts,
        required_case_ids=[r['id'] for r in records],paper_target=s.TARGETS[dataset]['cfm'])
    return group,records


@pytest.mark.parametrize('dataset',s.TARGETS)
def test_equal_seed_leaveout_weights_and_separate_uncertainty_units(dataset):
    group,records=fixture(dataset);r=s.summarize(records,group)
    assert r['training_seeds']==5 and r['degrees_of_freedom']==4
    assert r['independently_trained_models']==len(records)
    assert r['seed_mean_summary']['std']==pytest.approx(math.sqrt(.001/4))
    assert r['all_model_sample_STD']>r['seed_mean_summary']['std']
    assert not r['historical_paper_equivalence_certified'] and not r['author_reported_SD_aggregation_certified']


@pytest.mark.parametrize('dataset',s.TARGETS)
@pytest.mark.parametrize('fault',['one_leaveout','missing_seed','duplicate','wrong_dataset','wrong_method','mixed_recipe','nonfinite','wrong_identity'])
def test_no_partial_or_mixed_conditions_become_a_paper_row(dataset,fault):
    g,r=fixture(dataset)
    if fault=='one_leaveout':r=[x for x in r if x['leaveout_timepoint']==1]
    elif fault=='missing_seed':r=[x for x in r if x['seed']!=46]
    elif fault=='duplicate':r[-1]=deepcopy(r[0])
    elif fault=='wrong_dataset':r[0]['dataset']='other'
    elif fault=='wrong_method':r[0]['method']='otcfm'
    elif fault=='mixed_recipe':r[0]['protocol']='paper_declared'
    elif fault=='nonfinite':r[0]['heldout_W1']=float('nan')
    else:r[0]['id']='different'
    with pytest.raises(ValueError):s.summarize(r,g)


def test_registration_covers_all_original150_once_in_twelve_complete_rows():
    c=s.registration();ids=[i for g in c['groups'] for i in g['required_case_ids']]
    assert len(c['groups'])==12 and len(ids)==len(set(ids))==150
    assert sorted(len(g['required_case_ids']) for g in c['groups'])==[10]*6+[15]*6
    assert c['all_source_and_full_native_guards_required'] and c['no_table_result_from_partial_rows']
    assert c['settings']==s.read(s.ROOT/'configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json')['settings']


def test_controller_changes_only_the_twelve_group_scope():
    original=ast.parse(inspect.getsource(s.template.controller));changed=s.controller_tree()
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is int and n.value==12:n.value=6
            elif n.value=='all12_paper_row_outcomes_terminal_other_original_work_remains':n.value='all6_paper_row_outcomes_terminal_other_original_work_remains'
            return n
    assert ast.dump(original)==ast.dump(Reverse().visit(changed))


def test_real_complete_EB_native_report_matches_the_full_validation_contract():
    q=s.read(s.ROOT/s.native.QUEUE);rc=s.read(s.ROOT/s.replay.CONFIG);job=q['jobs'][0]
    source,proof=s.inspect(q,job);_,gp,rp=s.guard_paths(q,rc,job)
    report,guard=s.read(rp),s.read(gp)
    s.validate_native(job,source,proof,report,guard,s.sha(rp),s.sha(s.ROOT/s.replay.CONFIG))
    for mutation in ('partial_metric','wrong_matrix_shape','missing_timepoint','nonoptimal','objective_bound','extra_updates','failed_guard'):
        r,g=deepcopy(report),deepcopy(guard)
        if mutation=='partial_metric':r['native_metrics'].pop(next(iter(r['native_metrics'])))
        elif mutation=='wrong_matrix_shape':r['complete_interior_timepoint_cost_shapes'][0][0]-=1
        elif mutation=='missing_timepoint':r['full_test_population_distances_independently_recomputed'][0]['solves'].pop()
        elif mutation=='nonoptimal':r['full_test_population_distances_independently_recomputed'][0]['solves'][0]['native_LP_result_code']=0
        elif mutation=='objective_bound':r['full_test_population_distances_independently_recomputed'][0]['solves'][0]['objective_absolute_difference']=1.0
        elif mutation=='extra_updates':r['extra_model_training_updates']=1
        else:g['passed']=False
        with pytest.raises(ValueError):s.validate_native(job,source,proof,r,g,s.sha(rp),s.sha(s.ROOT/s.replay.CONFIG))
