from copy import deepcopy
import pytest

from scripts.flow_matching.otcfm_singlecell_native_contract_v1 import scientific_cases
from scripts.flow_matching.preflight_otcfm_singlecell_native_v2 import cases


def test_real_native_preflight_covers_both_complete_data_cohorts_methods_recipes_and_leftouts():
    jobs = scientific_cases(); selected = cases(dict(jobs=jobs))
    assert len(selected) == 24 and {j['seed'] for j in selected} == {42}
    assert len(jobs) == 120


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'changed_leftout', 'changed_recipe'])
def test_native_preflight_cannot_hide_an_unchecked_scientific_branch(fault):
    jobs = deepcopy(scientific_cases())
    if fault == 'missing': jobs.pop(0)
    elif fault == 'duplicate': jobs[5] = deepcopy(jobs[0])
    elif fault == 'changed_leftout': jobs[0]['leaveout_timepoint'] = 0
    else: jobs[0]['protocol'] = 'shortened_proxy'
    with pytest.raises(ValueError): cases(dict(jobs=jobs))


def test_registration_source_receipts_are_canonical_across_python_traversal_orders():
    from scripts.flow_matching.run_otcfm_singlecell_native_v2 import registration
    queue = registration()
    paths = [row['path'] for row in queue['source_receipts']]
    assert paths == sorted(set(paths))
    assert all(job['model_config'].endswith('.v2.json') for job in queue['jobs'])
    assert len(queue['jobs']) == 120
