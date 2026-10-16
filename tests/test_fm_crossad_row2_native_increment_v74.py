import csv
import json
from pathlib import Path
import pytest
from scripts.flow_matching import publish_crossad_row2_native_increment_v74 as s
from scripts.flow_matching import publish_native_condition_catalog_v55 as row


def test_declared_test_receipt_binds_the_actual_current_validation_module():
    assert (s.ROOT/s.TEST).resolve()==Path(__file__).resolve()
    receipts={r['path']:r['sha256'] for r in s.registration()['source_receipts']}
    assert receipts[s.TEST]==s.sha(Path(__file__))


def fixture():
    with (s.ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
    update=[dict(r,n='1',mean='.25',captured_utc='fixture_completion') for r in before if s.IDENTITY in r['protocol']]
    assert len(update)==15
    return before,update


def test_all_old_rows_and_capture_times_preserved_outside_actual_pending_case():
    before,update=fixture();after=s.merge(before,update)
    assert len(after)==11925 and sum(bool(x['mean']) for x in after)==3314
    assert sum(a!=b for a,b in zip(before,after))==15
    assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])


@pytest.mark.parametrize('fault',['missing_metric','duplicate','wrong_case','old_finite','old_undefined','invented_CI','author_claim'])
def test_no_incomplete_overwritten_or_unearned_result(fault):
    before,update=fixture()
    if fault=='missing_metric':update.pop()
    elif fault=='duplicate':update[-1]=dict(update[0])
    elif fault=='wrong_case':update[0]['score_recipe']=json.dumps(dict(canonical_id='other_case'))
    elif fault in ['old_finite','old_undefined']:
        i=next(i for i,x in enumerate(before) if row.key(x)==row.key(update[0]));before[i]['n']='1'
        if fault=='old_finite':before[i]['mean']='.5'
    elif fault=='invented_CI':update[0]['ci95_low']='.1'
    elif fault=='author_claim':update[0]['author_equivalence_certified']='True'
    with pytest.raises(ValueError):s.merge(before,update)
