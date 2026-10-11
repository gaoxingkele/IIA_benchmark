from copy import deepcopy

import pytest

from scripts.flow_matching.publish_pi_original_source_listing_v42 import ALGORITHM, METRICS, check_delta


def rows():
    before=[dict(algorithm='existing_method',captured_utc='original_capture',mean='1')]
    added=[dict(algorithm=ALGORITHM,dataset='PSM',n='1',required_repeats='6',protocol=p,metric=m,
        strict_TAB_result='False',author_equivalence_certified='False',std='',se='',ci95_low='',ci95_high='') for p,m in METRICS]
    return before,before+added


def test_only_new_original_seed_metrics_append_to_unchanged_history():
    before,after=rows()
    assert len(check_delta(before,after))==8
    changed=deepcopy(after)
    changed[0]['captured_utc']='recaptured'
    with pytest.raises(ValueError,match='prior metric'):
        check_delta(before,changed)


@pytest.mark.parametrize('key,value',[('n','2'),('required_repeats','1'),('std','0'),('strict_TAB_result','True')])
def test_one_replayed_seed_cannot_create_extra_seed_or_uncertainty_claim(key,value):
    before,after=rows()
    after[-1][key]=value
    with pytest.raises(ValueError,match='One original seed'):
        check_delta(before,after)


def test_threshold_protocols_cannot_merge_or_duplicate_source_seed():
    before,after=rows()
    altered=deepcopy(after)
    altered[-1]['protocol']='author_threshold_PA'
    with pytest.raises(ValueError,match='eight separate'):
        check_delta(before,altered)
    original=deepcopy(after[1])
    with pytest.raises(ValueError,match='count a canonical'):
        check_delta([original],[original]+after[1:])
