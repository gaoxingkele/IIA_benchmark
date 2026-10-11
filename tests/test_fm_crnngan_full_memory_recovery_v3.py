import copy
import pytest
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.register_crnngan_full_memory_recovery_v3 import recovery_queue, verify_scientific_equivalence


def test_all35_full_scientific_jobs_are_preserved_with_fresh_destinations():
    old = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    new = recovery_queue(old)
    assert new['flags'] == old['flags'] and new['variants'] == old['variants']
    assert len(new['preserved_prior_attempts']) == 35
    assert new['settings']['minimum_commit_headroom_bytes'] == 40*1024**3
    assert len({j['output_directory'] for j in new['jobs']}) == 35


def test_shortened_budget_and_overwrite_and_unreserved_memory_are_rejected():
    old = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    for kind in ['budget', 'destination', 'reserve']:
        bad = copy.deepcopy(recovery_queue(old))
        if kind == 'budget':
            bad['flags']['max_epoch'] = 100
        elif kind == 'destination':
            bad['jobs'][0]['output_directory'] = old['jobs'][0]['output_directory']
        else:
            bad['settings']['minimum_commit_headroom_bytes'] = 30*1024**3
        with pytest.raises(ValueError):
            verify_scientific_equivalence(old, bad)
