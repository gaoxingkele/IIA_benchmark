import csv
import json
import pytest
from scripts.flow_matching import publish_CrossAD25_Spectral10_native_increment_v70 as s
from scripts.flow_matching import publish_native_condition_catalog_v55 as rows


def fixture():
    with (s.ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
    updates=[]
    for r in before:
        if s.CROSS_ID in r['protocol'] or s.GEN_ID in r['protocol']:
            item=dict(r,n='1',mean='.25',captured_utc='fixture_completed')
            if s.GEN_ID in r['protocol']:item['evaluator_repeats']='5'
            updates.append(item)
    assert len(updates)==19
    return before,updates


def test_preserve_every_other_row_field_scope_and_capture_time():
    before,updates=fixture();after=s.merge(before,updates)
    assert len(after)==11925 and sum(bool(x['mean']) for x in after)==3254
    assert sum(a!=b for a,b in zip(before,after))==19
    assert all(a==b for a,b in zip(before,after) if s.CROSS_ID not in a['protocol'] and s.GEN_ID not in a['protocol'])


@pytest.mark.parametrize('fault',['missing_cross','missing_gen','duplicate','wrong_case','wrong_algorithm','short_evaluator',
    'old_finite','old_undefined','invented_training_CI','author_claim'])
def test_no_partial_overwritten_invented_or_shortened_result(fault):
    before,updates=fixture();cross=next(r for r in updates if s.CROSS_ID in r['protocol']);gen=next(r for r in updates if s.GEN_ID in r['protocol'])
    if fault=='missing_cross':updates.remove(cross)
    elif fault=='missing_gen':updates.remove(gen)
    elif fault=='duplicate':updates[-1]=dict(updates[0])
    elif fault=='wrong_case':cross['score_recipe']=json.dumps(dict(canonical_id='unverified'))
    elif fault=='wrong_algorithm':cross['algorithm']='CrossAD/full_model'
    elif fault=='short_evaluator':gen['evaluator_repeats']='4'
    elif fault in ['old_finite','old_undefined']:
        i=next(i for i,r in enumerate(before) if rows.key(r)==rows.key(cross));before[i]['n']='1'
        if fault=='old_finite':before[i]['mean']='.5'
    elif fault=='invented_training_CI':gen['ci95_low']='.1'
    else:cross['author_equivalence_certified']='True'
    with pytest.raises(ValueError):s.merge(before,updates)
