from copy import deepcopy
import itertools
import pytest
from scripts.flow_matching import analyze_otcfm_Multiome_complete120_comparisons_v1 as s


def fixture(method='cfm',shift=0):
    records=[dict(id=f'{method}_{seed}_{left}',dataset='fixture',protocol='released_yaml',method=method,seed=seed,
        leaveout_timepoint=left,heldout={m:1+(seed-42)*.01+(left-1)*.2+shift for m in s.METRICS})
        for seed,left in itertools.product(range(42,47),[1,2])]
    group=dict(dataset='fixture',protocol='released_yaml',method=method,required_case_ids=[r['id'] for r in records])
    return group,records


def test_exact_balanced_leaveout_seed_variance_decomposition_and_paired_seed_unit():
    g,r=fixture();v=s.balanced(r,g)['1-Wasserstein'];d=v['descriptive_variance_decomposition']
    assert d['total_SS']==pytest.approx(.102) and d['leaveout_SS']==pytest.approx(.1) and d['seed_SS']==pytest.approx(.002)
    assert d['residual_SS']==pytest.approx(0,abs=1e-28)
    assert d['leaveout_fraction']==pytest.approx(.1/.102)
    assert v['paired_leaveout2_minus1']['mean']==pytest.approx(.2)
    assert v['primary_five_seed_summary']['n']==5 and v['primary_five_seed_summary']['df']==4
    assert v['all10_model_sample_STD']>v['primary_five_seed_summary']['sample_STD']


def test_constant_values_do_not_fabricate_explained_variance():
    g,r=fixture()
    for row in r:row['heldout']={m:1.0 for m in s.METRICS}
    v=s.balanced(r,g)['1-Wasserstein']['descriptive_variance_decomposition']
    assert v['total_SS']==0 and v['leaveout_fraction'] is None and v['seed_fraction'] is None


@pytest.mark.parametrize('fault',['missing','duplicate','recipe','dataset','method','nonfinite','partial_metrics'])
def test_no_incomplete_or_mixed_conditions_are_summarized(fault):
    g,r=fixture()
    if fault=='missing':r=r[:-1]
    elif fault=='duplicate':r[-1]=deepcopy(r[0])
    elif fault in ['recipe','dataset','method']:r[0][{'recipe':'protocol','dataset':'dataset','method':'method'}[fault]]='wrong'
    elif fault=='nonfinite':r[0]['heldout']['1-Wasserstein']=float('nan')
    else:r[0]['heldout'].pop('1-Wasserstein')
    with pytest.raises(ValueError):s.balanced(r,g)


def test_method_pairs_match_five_seed_level_leaveout_means_and_keep_direction():
    a,ar=fixture();b,br=fixture('otcfm',-.05)
    r=s.paired(ar,br,a,b)['1-Wasserstein']
    assert r['paired_five_seed_summary']['mean']==pytest.approx(-.05)
    assert r['negative_difference_favors_methodB'] and r['paired_five_seed_summary']['n']==5
    b['protocol']='different'
    with pytest.raises(ValueError):s.paired(ar,br,a,b)


def test_all120_models_are_referenced_without_360_pseudo_independent_training_credits():
    c=s.registration();ids=[i for g in c['groups'] for i in g['required_case_ids']]
    assert len(c['groups'])==24 and len(set(ids))==120 and len(ids)==360
    assert sum(g['kind']=='within' for g in c['groups'])==sum(g['kind']=='paired' for g in c['groups'])==12
    assert c['posthoc_diagnostics'] and not c['hypothesis_testing_or_causal_claim']
