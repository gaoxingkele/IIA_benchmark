from copy import deepcopy
import itertools
import math
import pytest
from scripts.flow_matching import summarize_otcfm_EB100_full_paper_rows_v1 as s


def rows():
    return [dict(id=f'fixture_{seed}_{left}',method='cfm',protocol='released_yaml',seed=seed,
        leaveout_timepoint=left,heldout_W1=seed/100+left/10) for seed,left in itertools.product(range(42,47),[1,2,3])]


def test_equal_three_leaveout_then_five_seed_mean_with_distinct_uncertainty_units():
    r=s.summarize(rows(),'cfm','released_yaml')
    assert r['independently_trained_models']==15 and r['training_seeds']==5 and r['degrees_of_freedom']==4
    assert r['seed_level_leaveout_means']==pytest.approx([.62,.63,.64,.65,.66])
    assert r['seed_mean_summary']['mean']==pytest.approx(.64)
    assert r['seed_mean_summary']['std']==pytest.approx(math.sqrt(.001/4))
    assert r['all15_model_sample_STD']>r['seed_mean_summary']['std']
    assert not r['historical_paper_equivalence_certified'] and not r['author_reported_SD_aggregation_certified']


@pytest.mark.parametrize('fault',['single_model','missing_leaveout','missing_seed','duplicate','wrong_method','mixed_recipe','nonfinite'])
def test_partial_or_mixed_conditions_never_become_a_paper_row(fault):
    a=deepcopy(rows())
    if fault=='single_model':a=a[:1]
    elif fault=='missing_leaveout':a=[r for r in a if r['leaveout_timepoint']!=3]
    elif fault=='missing_seed':a=[r for r in a if r['seed']!=46]
    elif fault=='duplicate':a[-1]=deepcopy(a[0])
    elif fault=='wrong_method':a[0]['method']='otcfm'
    elif fault=='mixed_recipe':a[0]['protocol']='paper_declared'
    else:a[0]['heldout_W1']=float('nan')
    with pytest.raises(ValueError):s.summarize(a,'cfm','released_yaml')


def test_every_original90_native_case_belongs_to_exactly_one_of_six_complete_rows():
    c=s.registration();ids=[i for g in c['groups'] for i in g['required_case_ids']]
    assert c['required_rows']==len(c['groups'])==6 and len(ids)==len(set(ids))==90
    assert all(len(g['required_case_ids'])==15 for g in c['groups'])
    assert c['all_15_models_and_native_replays_required'] and c['no_table_result_from_partial_rows']
    assert c['settings']==s.read(s.ROOT/'configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json')['settings']
