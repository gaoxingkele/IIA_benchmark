import copy

import pytest

from scripts.flow_matching.audit_spectral_terminal_sampling_checkpoint_v2 import selected_job, terminal_failure


def fixture():
    job = dict(id='energy42', model='spectral_flow', dataset='energy', seed=42,
        training_updates=25000, data_shape=[19712, 24, 28], generation_batch_size=2001,
        generation_field_chunk_size=None, expected_parameters=1086432)
    queue = {'jobs': [job] + [dict(id=str(i)) for i in range(69)]}
    return queue, {'original_job': copy.deepcopy(job)}, job


@pytest.mark.parametrize('field,value', [('training_updates', 10000), ('data_shape', [100, 24, 28]),
    ('generation_batch_size', 32), ('generation_field_chunk_size', 1),
    ('expected_parameters', 356800), ('seed', 43)])
def test_smaller_or_changed_job_cannot_satisfy_frozen_full_attempt(field, value):
    queue, config, job = fixture()
    assert selected_job(queue, config) == job
    queue['jobs'][0][field] = value
    with pytest.raises(ValueError):
        selected_job(queue, config)


@pytest.mark.parametrize('fault', ['removed_job', 'duplicate_id', 'live_result', 'generated',
    'partial_training', 'successful_exit', 'validated', 'no_resource_abort', 'wrong_id'])
def test_incomplete_registry_or_nonterminal_failure_is_rejected(fault):
    queue, config, job = fixture()
    progress = dict(phase='sampling', optimizer_updates=25000)
    receipt = dict(id=job['id'], exitcode=1, full_result_validated=False, resource_guard_aborted=True)
    assert selected_job(queue, config) == job
    terminal_failure(job, progress, receipt, False, False)
    if fault == 'removed_job':
        queue['jobs'].pop()
    elif fault == 'duplicate_id':
        queue['jobs'][-1] = copy.deepcopy(job)
    elif fault == 'partial_training':
        progress['optimizer_updates'] = 24900
    elif fault == 'successful_exit':
        receipt['exitcode'] = 0
    elif fault == 'validated':
        receipt['full_result_validated'] = True
    elif fault == 'no_resource_abort':
        receipt['resource_guard_aborted'] = False
    elif fault == 'wrong_id':
        receipt['id'] = 'other'
    with pytest.raises(ValueError):
        selected_job(queue, config)
        terminal_failure(job, progress, receipt, fault == 'live_result', fault == 'generated')
