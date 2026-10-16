import csv
import json
import pytest
from scripts.flow_matching import publish_crossad_two_native_increments_v63 as s
from scripts.flow_matching import publish_native_condition_catalog_v55 as p


def fixtures():
    with (s.ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        before=list(csv.DictReader(f))
    updates=[]
    for row in before:
        if row['task']=='anomaly_detection' and any(cid in row['protocol'] for cid in s.CASES):
            assert row['n']=='0' and not row['mean']
            updates.append(dict(row,n='1',mean='.123',captured_utc='test_full_completion'))
    assert len(updates)==30
    return before,updates


def test_exact_thirty_pending_controls_and_every_other_row_and_time_preserved():
    before,updates=fixtures();after=s.merge(before,updates);changed={p.key(x) for x in updates}
    assert len(after)==len(before)==11925 and sum(bool(x['mean']) for x in after)==3156
    assert sum(a!=b for a,b in zip(before,after))==30
    assert all(a==b for a,b in zip(before,after) if p.key(a) not in changed)


@pytest.mark.parametrize('fault',['missing_control','duplicate','wrong_case','finite_existing','finished_undefined','CI','author_certificate','short_base'])
def test_no_partial_case_overwrite_or_unearned_uncertainty(fault):
    before,updates=fixtures()
    if fault=='missing_control':updates.pop()
    elif fault=='duplicate':updates[-1]=dict(updates[0])
    elif fault=='wrong_case':updates[0]['score_recipe']=json.dumps(dict(canonical_id='unverified_case'))
    elif fault in ('finite_existing','finished_undefined'):
        i=next(i for i,r in enumerate(before) if p.key(r)==p.key(updates[0]));before[i]['n']='1'
        if fault=='finite_existing':before[i]['mean']='.5'
    elif fault=='CI':updates[0]['ci95_low']='.01'
    elif fault=='author_certificate':updates[0]['author_equivalence_certified']='True'
    elif fault=='short_base':before.pop()
    with pytest.raises(ValueError):s.merge(before,updates)
