import copy

import pytest

from scripts.flow_matching import run_selected_incident_source_recovery_v6 as selected


@pytest.mark.parametrize('fault', [None, 'branch', 'kind', 'canonical', 'duplicate'])
def test_source_retry_selects_one_original_case_without_changing_full_registry(fault, monkeypatch):
    cases = [dict(kind='pi', original_id=str(i), job=dict(id='retry'+str(i), branch='source_runtime'))
             for i in range(13)]
    registration = dict(selected_job_id='retry7', selected_original_id='7', resource_lock='original-mutex')
    if fault == 'branch': cases[7]['job']['branch'] = 'causal_axis_repaired'
    elif fault == 'kind': cases[7]['kind'] = 'crossad'
    elif fault == 'canonical': cases[7]['original_id'] = 'other'
    elif fault == 'duplicate': cases[8] = copy.deepcopy(cases[7])
    snapshot = copy.deepcopy(cases)
    namespace = {}
    exec('def original_main(): pass', namespace)
    monkeypatch.setattr(selected, 'bind', lambda config: (namespace['original_main'],
        dict(jobs=cases), dict(source_path='scripts/flow_matching/run_resource_recovery.py')))
    if fault:
        with pytest.raises(ValueError): selected.bind_source(registration)
    else:
        function, queue, binding = selected.bind_source(registration)
        assert len(queue['jobs']) == 13 and function.__globals__['selected_job_id'] == 'retry7'
    assert cases == snapshot
