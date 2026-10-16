import csv
import json
import pytest
from scripts.flow_matching import publish_spectral_energy45_native_increment_v66 as s
from scripts.flow_matching import publish_native_condition_catalog_v55 as row


def fixture():
    with (s.ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
    update=[dict(r,n='1',mean='.25',evaluator_repeats='5',captured_utc='fixture_completion') for r in before if s.IDENTITY in r['protocol']]
    assert len(update)==4
    return before,update


def test_preserve_every_other_row_and_original_capture_time():
    before,update=fixture();after=s.merge(before,update)
    assert len(after)==11925 and sum(bool(r['mean']) for r in after)==3190
    assert sum(a!=b for a,b in zip(before,after))==4
    assert all(a==b for a,b in zip(before,after) if s.IDENTITY not in a['protocol'])


@pytest.mark.parametrize('fault',['missing_metric','wrong_case','old_finite','old_undefined','invented_training_CI','short_evaluator','author_claim'])
def test_no_incomplete_overwritten_or_overclaimed_metric(fault):
    before,update=fixture()
    if fault=='missing_metric':update.pop()
    elif fault=='wrong_case':update[0]['score_recipe']=json.dumps(dict(canonical_id='other'))
    elif fault in ['old_finite','old_undefined']:
        i=next(i for i,r in enumerate(before) if row.key(r)==row.key(update[0]));before[i]['n']='1'
        if fault=='old_finite':before[i]['mean']='.5'
    elif fault=='invented_training_CI':update[0]['ci95_low']='.1'
    elif fault=='short_evaluator':update[0]['evaluator_repeats']='4'
    else:update[0]['author_equivalence_certified']='True'
    with pytest.raises(ValueError):s.merge(before,update)
