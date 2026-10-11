from copy import deepcopy

import pytest

from scripts.flow_matching.publish_grasp_tau025_two_seed_listing_v45 import PROFILES,PROTOCOLS,check_delta


def rows():
    before=[dict(algorithm='unrelated',captured_utc='original_time',mean='1')]
    for profile in PROFILES:
        for protocol,_,names in PROTOCOLS:
            for name in names:
                before.append(dict(algorithm='GRASP/'+profile,dataset='swan',protocol=protocol,metric=name,
                    n='1',required_repeats='10',mean='.5',captured_utc='original_time',source='original'))
    after=deepcopy(before)
    for row in after[1:]:
        row.update(n='2',mean='.6',captured_utc='new_time',source='new_proof')
    return before,after


def test_exact_tau025_partial_cohort_advance_without_recapturing_other_results():
    before,after=rows()
    assert len(check_delta(before,after))==6
    after[0]['captured_utc']='changed'
    with pytest.raises(ValueError):
        check_delta(before,after)


@pytest.mark.parametrize('key,value',[('required_repeats','2'),('dataset','other'),('protocol','merged_PA'),('n','10')])
def test_original_seed_obligation_dataset_and_protocol_cannot_be_shrunk(key,value):
    before,after=rows()
    after[1][key]=value
    with pytest.raises(ValueError):
        check_delta(before,after)


def test_all_six_metrics_of_the_tau025_cohort_and_all_old_slots_are_required():
    before,after=rows()
    after[-1]=deepcopy(before[-1])
    with pytest.raises(ValueError,match='Exactly all six'):
        check_delta(before,after)
    with pytest.raises(ValueError,match='Every existing metric'):
        check_delta(before,after[:-1])
